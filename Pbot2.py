#!/usr/bin/env python3
"""
StealthVolume CLI - Jupiter API v6 Integration with Advanced AMM
Production-ready implementation with all latest features
"""

import asyncio
import base64
import json
import logging
import random
import time
import yaml
import aiohttp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.transaction import VersionedTransaction
from solders.system_program import TransferParams, transfer
from solders.message import Message
import base58
from cryptography.fernet import Fernet
import click
import os
import sys
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum
import statistics

# Load environment variables first
load_dotenv()

# =============================================================================
# ENHANCED JUPITER API v6 CLIENT
# =============================================================================

class JupiterV6Client:
    """Enhanced Jupiter API v6 client with all latest features"""
    
    def __init__(self, session: aiohttp.ClientSession, config: Any):
        self.session = session
        self.config = config
        self.base_url = "https://quote-api.jup.ag/v6"
        self.last_quote_time = 0
        self.quote_cache = {}
        
    async def get_quote_v6(self, 
                          input_mint: str, 
                          output_mint: str, 
                          amount: int,
                          slippage_bps: int = 50,
                          swap_mode: str = "ExactIn",
                          dexes: List[str] = None,
                          only_direct_routes: bool = False,
                          as_legacy_transaction: bool = False,
                          max_accounts: int = 64) -> Dict[str, Any]:
        """
        Get quote from Jupiter API v6 with enhanced parameters
        https://station.jup.ag/docs/apis/swap-api#get-quote
        """
        try:
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': amount,
                'slippageBps': slippage_bps,
                'swapMode': swap_mode,
                'asLegacyTransaction': str(as_legacy_transaction).lower(),
                'maxAccounts': str(max_accounts)
            }
            
            if dexes:
                params['dexes'] = ','.join(dexes)
            if only_direct_routes:
                params['onlyDirectRoutes'] = 'true'
            
            # Rate limiting protection
            current_time = time.time()
            if current_time - self.last_quote_time < 0.1:  # 100ms minimum between quotes
                await asyncio.sleep(0.1)
            
            async with self.session.get(f"{self.base_url}/quote", params=params) as response:
                self.last_quote_time = time.time()
                
                if response.status == 200:
                    quote_data = await response.json()
                    logger.info(f"Quote obtained: {quote_data.get('outAmount', 0)/1e9:.6f} output")
                    return quote_data
                else:
                    error_text = await response.text()
                    logger.error(f"Jupiter API v6 quote error {response.status}: {error_text}")
                    raise Exception(f"Jupiter API v6 quote error: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error during quote request: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during quote request: {e}")
            raise
    
    async def swap_v6(self, 
                     quote_response: Dict[str, Any], 
                     wallet: Keypair,
                     wrap_and_unwrap_sol: bool = True,
                     dynamic_compute_unit_limit: bool = True,
                     prioritization_fee_lamports: str = 'auto',
                     use_shared_accounts: bool = True,
                     as_legacy_transaction: bool = False) -> Dict[str, Any]:
        """
        Execute swap using Jupiter API v6 with enhanced features
        https://station.jup.ag/docs/apis/swap-api#post-swap
        """
        try:
            payload = {
                'quoteResponse': quote_response,
                'userPublicKey': str(wallet.pubkey()),
                'wrapAndUnwrapSol': wrap_and_unwrap_sol,
                'dynamicComputeUnitLimit': dynamic_compute_unit_limit,
                'prioritizationFeeLamports': prioritization_fee_lamports,
                'useSharedAccounts': use_shared_accounts,
                'asLegacyTransaction': as_legacy_transaction
            }
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            async with self.session.post(f"{self.base_url}/swap", json=payload, headers=headers) as response:
                if response.status == 200:
                    swap_data = await response.json()
                    
                    # Handle different response formats
                    if 'swapTransaction' in swap_data:
                        return swap_data
                    elif 'error' in swap_data:
                        logger.error(f"Swap response error: {swap_data['error']}")
                        return swap_data
                    else:
                        logger.error("Unexpected swap response format")
                        return {'error': 'Unexpected response format'}
                else:
                    error_text = await response.text()
                    logger.error(f"Swap execution failed {response.status}: {error_text}")
                    return {'error': f'HTTP {response.status}: {error_text}'}
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error during swap execution: {e}")
            return {'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during swap execution: {e}")
            return {'error': str(e)}
    
    async def execute_swap_transaction(self, swap_data: Dict[str, Any], wallet: Keypair) -> bool:
        """Execute the swap transaction on Solana"""
        try:
            if 'error' in swap_data:
                logger.error(f"Cannot execute swap due to error: {swap_data['error']}")
                return False
            
            swap_transaction = swap_data.get('swapTransaction')
            if not swap_transaction:
                logger.error("No swap transaction in response")
                return False
            
            # Decode and send transaction
            tx_bytes = base64.b64decode(swap_transaction)
            transaction = VersionedTransaction.from_bytes(tx_bytes)
            
            async with AsyncClient(self.config.rpc_url) as client:
                # Simulate first to check for errors
                simulate_result = await client.simulate_transaction(transaction)
                if simulate_result.value and simulate_result.value.err:
                    logger.error(f"Transaction simulation failed: {simulate_result.value.err}")
                    return False
                
                # Send actual transaction
                result = await client.send_transaction(transaction, wallet)
                if result.value:
                    logger.info(f"Swap transaction sent: {result.value}")
                    # Confirm transaction
                    confirmation = await client.confirm_transaction(result.value, commitment=Confirmed)
                    if confirmation.value:
                        logger.info("Swap transaction confirmed")
                        return True
                    else:
                        logger.error("Swap transaction failed confirmation")
                        return False
                else:
                    logger.error("Failed to send transaction")
                    return False
                    
        except Exception as e:
            logger.error(f"Transaction execution error: {e}")
            return False
    
    async def get_price_v6(self, ids: str, vs_token: str = "USDC") -> Dict[str, Any]:
        """Get token prices from Jupiter Price API"""
        try:
            url = f"https://price.jup.ag/v6/price"
            params = {'ids': ids, 'vsToken': vs_token}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Price API error {response.status}: {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"Price API request failed: {e}")
            return {}
    
    async def get_tokens_v6(self) -> List[Dict[str, Any]]:
        """Get tokens list from Jupiter API"""
        try:
            url = "https://tokens.jup.ag/tokens"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
        except Exception as e:
            logger.error(f"Tokens API request failed: {e}")
            return []
    
    async def get_swap_instructions(self, quote_response: Dict[str, Any], wallet: Keypair) -> List[Any]:
        """Get swap instructions for advanced transaction building"""
        try:
            swap_data = await self.swap_v6(quote_response, wallet)
            if 'swapTransaction' not in swap_data:
                return []
            
            tx_bytes = base64.b64decode(swap_data['swapTransaction'])
            transaction = VersionedTransaction.from_bytes(tx_bytes)
            return transaction.message.instructions
        except Exception as e:
            logger.error(f"Failed to get swap instructions: {e}")
            return []

# =============================================================================
# ENHANCED AMM WITH JUPITER INTEGRATION
# =============================================================================

class AdvancedAMMWithJupiter:
    """Advanced AMM that uses Jupiter for optimal routing"""
    
    def __init__(self, config: Any, jupiter_client: JupiterV6Client):
        self.config = config
        self.jupiter = jupiter_client
        self.inventory = {
            'base_token': config.amm_config.base_token_reserve,
            'quote_token': config.amm_config.quote_token_reserve
        }
        self.price_history = []
        self.trade_history = []
        self.current_price = config.amm_config.initial_price
        
    async def calculate_optimal_swap(self, trade_type: str, amount: float) -> Tuple[float, float, Dict[str, Any]]:
        """Calculate optimal swap using Jupiter for pricing"""
        try:
            # Convert to lamports
            amount_lamports = int(amount * 1e9)
            
            # Determine mint addresses based on trade type
            if trade_type == "buy":
                input_mint = "So11111111111111111111111111111111111111112"  # SOL
                output_mint = self.config.token_address
            else:  # sell
                input_mint = self.config.token_address
                output_mint = "So11111111111111111111111111111111111111112"  # SOL
            
            # Get quote from Jupiter
            quote = await self.jupiter.get_quote_v6(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount_lamports,
                slippage_bps=int(self.config.amm_config.max_slippage * 100),
                dexes=self.config.dexes if hasattr(self.config, 'dexes') else None
            )
            
            if not quote:
                return self._fallback_pricing(trade_type, amount)
            
            # Calculate effective price
            if trade_type == "buy":
                output_amount = int(quote.get('outAmount', 0))
                effective_price = amount_lamports / output_amount if output_amount > 0 else 0
            else:
                input_amount = int(quote.get('inAmount', 0))
                output_amount = int(quote.get('outAmount', 0))
                effective_price = output_amount / input_amount if input_amount > 0 else 0
            
            slippage = self._calculate_jupiter_slippage(quote)
            
            return effective_price, slippage, quote
            
        except Exception as e:
            logger.error(f"Optimal swap calculation failed: {e}")
            return self._fallback_pricing(trade_type, amount)
    
    def _fallback_pricing(self, trade_type: str, amount: float) -> Tuple[float, float, Dict[str, Any]]:
        """Fallback pricing when Jupiter is unavailable"""
        if trade_type == "buy":
            price = self.current_price * (1 + self.config.amm_config.spread_target / 2)
        else:
            price = self.current_price * (1 - self.config.amm_config.spread_target / 2)
        
        slippage = self.config.amm_config.max_slippage / 100
        return price, slippage, {}
    
    def _calculate_jupiter_slippage(self, quote: Dict[str, Any]) -> float:
        """Calculate slippage from Jupiter quote"""
        try:
            price_impact_pct = quote.get('priceImpactPct', 0)
            if isinstance(price_impact_pct, (int, float)):
                return abs(price_impact_pct)
            return self.config.amm_config.max_slippage / 100
        except:
            return self.config.amm_config.max_slippage / 100
    
    async def execute_amm_trade(self, wallet: Keypair, trade_type: str, amount: float) -> bool:
        """Execute AMM trade using Jupiter for optimal execution"""
        try:
            # Calculate optimal swap
            price, slippage, quote = await self.calculate_optimal_swap(trade_type, amount)
            
            if not quote:
                logger.error("No quote available for AMM trade")
                return False
            
            # Execute swap through Jupiter
            swap_data = await self.jupiter.swap_v6(quote, wallet)
            
            if 'error' in swap_data:
                logger.error(f"AMM swap failed: {swap_data['error']}")
                return False
            
            # Execute transaction
            success = await self.jupiter.execute_swap_transaction(swap_data, wallet)
            
            if success:
                # Update AMM inventory
                self._update_inventory(trade_type, amount, price)
                self._record_trade(trade_type, amount, price, slippage)
                logger.info(f"AMM {trade_type} executed: {amount:.4f} at price {price:.6f}")
            
            return success
            
        except Exception as e:
            logger.error(f"AMM trade execution failed: {e}")
            return False
    
    def _update_inventory(self, trade_type: str, amount: float, price: float):
        """Update AMM inventory after trade"""
        if trade_type == "buy":
            self.inventory['quote_token'] += amount * price
            self.inventory['base_token'] -= amount
        else:  # sell
            self.inventory['quote_token'] -= amount * price
            self.inventory['base_token'] += amount
        
        # Update current price based on inventory ratio
        if self.inventory['base_token'] > 0:
            self.current_price = self.inventory['quote_token'] / self.inventory['base_token']
        
        self.price_history.append(self.current_price)
        # Keep only recent history
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
    
    def _record_trade(self, trade_type: str, amount: float, price: float, slippage: float):
        """Record trade for analysis"""
        trade_data = {
            'timestamp': time.time(),
            'type': trade_type,
            'amount': amount,
            'price': price,
            'slippage': slippage,
            'inventory_ratio': self.inventory['base_token'] / (self.inventory['base_token'] + self.inventory['quote_token'] / price)
        }
        self.trade_history.append(trade_data)
        
        # Keep only recent history
        if len(self.trade_history) > 10000:
            self.trade_history = self.trade_history[-10000:]

# =============================================================================
# ENHANCED TRADING ENGINE WITH JUPITER v6
# =============================================================================

class JupiterV6TradingEngine:
    """Enhanced trading engine with Jupiter v6 integration"""
    
    def __init__(self, config: Any):
        self.config = config
        self.jupiter_client = None
        self.amm = None
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_volume': 0.0,
            'jupiter_swaps': 0,
            'amm_trades': 0,
            'start_time': time.time()
        }
    
    async def initialize(self):
        """Initialize trading engine with Jupiter v6"""
        self.jupiter_client = JupiterV6Client(aiohttp.ClientSession(), self.config)
        self.amm = AdvancedAMMWithJupiter(self.config, self.jupiter_client)
        logger.info("Jupiter v6 trading engine initialized")
    
    async def execute_advanced_swap(self, 
                                  wallet: Keypair,
                                  input_mint: str,
                                  output_mint: str,
                                  amount: float,
                                  strategy: str = "optimal") -> bool:
        """Execute advanced swap with strategy selection"""
        try:
            amount_lamports = int(amount * 1e9)
            
            # Get quote based on strategy
            if strategy == "fast":
                dexes = ["Raydium"]  # Fastest DEX
            elif strategy == "cheap":
                dexes = ["Orca", "Raydium"]  # Lowest fees
            else:  # optimal
                dexes = self.config.dexes
            
            quote = await self.jupiter_client.get_quote_v6(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount_lamports,
                slippage_bps=int(self.config.ai_parameters.max_slippage * 100),
                dexes=dexes
            )
            
            if not quote:
                return False
            
            # Execute swap
            swap_data = await self.jupiter_client.swap_v6(quote, wallet)
            success = await self.jupiter_client.execute_swap_transaction(swap_data, wallet)
            
            if success:
                self.performance_metrics['total_trades'] += 1
                self.performance_metrics['successful_trades'] += 1
                self.performance_metrics['jupiter_swaps'] += 1
                self.performance_metrics['total_volume'] += amount
            
            return success
            
        except Exception as e:
            logger.error(f"Advanced swap execution failed: {e}")
            return False
    
    async def run_hybrid_market_making(self, duration_hours: int = 24):
        """Run hybrid market making using both AMM and Jupiter"""
        logger.info(f"Starting hybrid market making for {duration_hours} hours")
        
        end_time = time.time() + duration_hours * 3600
        wallets = await load_wallets(self.config)  # Your existing wallet loading function
        
        if not wallets:
            logger.error("No wallets available")
            return
        
        iteration = 0
        
        while time.time() < end_time:
            try:
                wallet = random.choice(wallets)
                
                # Decide between AMM and Jupiter swap
                use_amm = random.random() < 0.3  # 30% AMM, 70% Jupiter
                
                if use_amm and self.config.strategies.amm['enabled']:
                    # Execute AMM trade
                    trade_type = random.choice(["buy", "sell"])
                    amount = random.uniform(
                        self.config.amm_config.min_trade_size,
                        self.config.amm_config.max_trade_size
                    )
                    
                    success = await self.amm.execute_amm_trade(wallet, trade_type, amount)
                    
                    if success:
                        self.performance_metrics['amm_trades'] += 1
                
                else:
                    # Execute Jupiter swap
                    input_mint = "So11111111111111111111111111111111111111112"  # SOL
                    output_mint = self.config.token_address
                    amount = random.uniform(0.01, 0.5)
                    
                    success = await self.execute_advanced_swap(
                        wallet, input_mint, output_mint, amount
                    )
                
                # Adaptive delay based on market conditions
                delay = random.uniform(5, 30)
                await asyncio.sleep(delay)
                
                iteration += 1
                if iteration % 10 == 0:
                    self.log_performance()
                    
            except Exception as e:
                logger.error(f"Hybrid market making iteration failed: {e}")
                await asyncio.sleep(5)
    
    def log_performance(self):
        """Log performance metrics"""
        total_trades = self.performance_metrics['total_trades']
        successful_trades = self.performance_metrics['successful_trades']
        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        logger.info(
            f"Performance: {total_trades} total trades, "
            f"{success_rate:.1%} success rate, "
            f"{self.performance_metrics['jupiter_swaps']} Jupiter swaps, "
            f"{self.performance_metrics['amm_trades']} AMM trades"
        )

# =============================================================================
# ENHANCED CLI COMMANDS
# =============================================================================

@click.group()
@click.version_option(version='3.0.0')
def cli():
    """StealthVolume CLI v3 - Jupiter v6 with Advanced AMM"""
    pass

@cli.command()
@click.option('--config-source', default='auto', type=click.Choice(['auto', 'yaml', 'env']))
@click.option('--config-path', default='config.yaml')
@click.option('--input-mint', required=True, help='Input token mint address')
@click.option('--output-mint', required=True, help='Output token mint address')
@click.option('--amount', type=float, required=True, help='Amount to swap')
@click.option('--slippage', type=float, default=1.0, help='Slippage percentage')
@click.option('--strategy', type=click.Choice(['optimal', 'fast', 'cheap']), default='optimal')
def swap(config_source: str, config_path: str, input_mint: str, output_mint: str, 
         amount: float, slippage: float, strategy: str):
    """Execute a swap using Jupiter v6 API"""
    config = load_config(config_source, config_path)
    engine = JupiterV6TradingEngine(config)
    
    async def run_swap():
        await engine.initialize()
        wallets = await load_wallets(config)
        if not wallets:
            logger.error("No wallets available")
            return
        
        wallet = wallets[0]  # Use first wallet
        success = await engine.execute_advanced_swap(
            wallet, input_mint, output_mint, amount, strategy
        )
        
        if success:
            logger.info("Swap executed successfully")
        else:
            logger.error("Swap failed")
    
    asyncio.run(run_swap())

@cli.command()
@click.option('--config-source', default='auto', type=click.Choice(['auto', 'yaml', 'env']))
@click.option('--config-path', default='config.yaml')
@click.option('--duration', default=24, help='Duration in hours')
def hybrid_mm(config_source: str, config_path: str, duration: int):
    """Run hybrid market making with Jupiter v6 and AMM"""
    config = load_config(config_source, config_path)
    engine = JupiterV6TradingEngine(config)
    
    try:
        asyncio.run(engine.initialize())
        asyncio.run(engine.run_hybrid_market_making(duration))
    except KeyboardInterrupt:
        logger.info("Hybrid market making stopped by user")
        engine.log_performance()
    except Exception as e:
        logger.error(f"Hybrid market making failed: {e}")

@cli.command()
@click.option('--config-source', default='auto', type=click.Choice(['auto', 'yaml', 'env']))
@click.option('--config-path', default='config.yaml')
@click.option('--ids', required=True, help='Comma-separated token IDs')
@click.option('--vs-token', default='USDC', help='Vs token for pricing')
def price(config_source: str, config_path: str, ids: str, vs_token: str):
    """Get token prices from Jupiter Price API"""
    config = load_config(config_source, config_path)
    engine = JupiterV6TradingEngine(config)
    
    async def get_prices():
        await engine.initialize()
        prices = await engine.jupiter_client.get_price_v6(ids, vs_token)
        if prices:
            logger.info(f"Price data: {json.dumps(prices, indent=2)}")
        else:
            logger.error("Failed to get price data")
    
    asyncio.run(get_prices())

if __name__ == '__main__':
    print("=" * 80)
    print("🎯 STEALTHVOLUME CLI v3 - JUPITER v6 WITH ADVANCED AMM")
    print("=" * 80)
    print("FEATURES:")
    print("  • Jupiter API v6 Full Integration")
    print("  • Hybrid Market Making (AMM + Jupiter)")
    print("  • Advanced Swap Strategies")
    print("  • Real-time Price Feeds")
    print("  • Comprehensive Configuration System")
    print("=" * 80)
    print()
    
    cli()
