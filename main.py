import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List

# ==========================================
# 1. DATA SCHEMAS
# ==========================================
class SessionTelemetry(BaseModel):
    user_id: str
    clickstream_intent_score: float = Field(..., ge=0.0, le=1.0)
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    current_cart_value: float

class MarketContext(BaseModel):
    sku_id: str
    competitor_avg_price: float
    demand_surge_factor: float
    inventory_count: int

class StrategyDecision(BaseModel):
    sku_id: str
    adjusted_price: float
    recommended_bundle: Optional[List[str]] = None
    discount_incentive_pct: float = 0.0

# ==========================================
# 2. SPECIALIZED AGENTS
# ==========================================
class MarketIntelligenceAgent:
    def analyze(self, market: MarketContext) -> dict:
        elasticity_index = round(float(np.clip(1.0 / (market.demand_surge_factor + 1e-5), 0.2, 2.5)), 3)
        inventory_risk = "LOW" if market.inventory_count > 50 else ("HIGH" if market.inventory_count < 10 else "MODERATE")
        target_ceiling = market.competitor_avg_price * (1.15 if market.demand_surge_factor > 1.2 else 0.95)
        return {
            "elasticity_index": elasticity_index,
            "inventory_risk": inventory_risk,
            "recommended_base_ceiling": target_ceiling
        }

class PersonaBehaviorAgent:
    def evaluate(self, telemetry: SessionTelemetry) -> dict:
        intent_weight = telemetry.clickstream_intent_score * 0.7
        churn_penalty = telemetry.churn_probability * 0.3
        composite_affinity = round(float(intent_weight - churn_penalty), 3)
        eligible_for_discount = telemetry.churn_probability > 0.6 and telemetry.clickstream_intent_score > 0.4

        return {
            "composite_affinity": composite_affinity,
            "eligible_for_discount": eligible_for_discount,
            "max_allowable_discount_pct": 15.0 if eligible_for_discount else 0.0
        }

class StrategyPolicyAgent:
    def arbitrate(self, market: MarketContext, telemetry: SessionTelemetry, 
                  market_analysis: dict, persona_analysis: dict, policy_bias: float = 1.0) -> StrategyDecision:
        base_target = market_analysis["recommended_base_ceiling"] * policy_bias
        discount = persona_analysis["max_allowable_discount_pct"]
        optimized_price = round(base_target * (1.0 - (discount / 100.0)), 2)
        
        bundle = None
        if persona_analysis["composite_affinity"] > 0.3 and market.inventory_count > 30:
            bundle = [f"{market.sku_id}-ACC-01", f"{market.sku_id}-CARE-PLUS"]

        return StrategyDecision(
            sku_id=market.sku_id,
            adjusted_price=optimized_price,
            recommended_bundle=bundle,
            discount_incentive_pct=discount
        )

class MetaCriticAgent:
    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate

    def evaluate_and_evolve(self, ground_truth_converted: bool, current_policy_bias: float) -> tuple[float, float]:
        if ground_truth_converted:
            reward = 1.0
            new_policy_bias = current_policy_bias + (self.learning_rate * 0.5)
        else:
            reward = -1.0
            new_policy_bias = current_policy_bias - self.learning_rate

        clamped_bias = float(np.clip(new_policy_bias, 0.80, 1.30))
        return reward, round(clamped_bias, 4)

# ==========================================
# 3. RUNNABLE SIMULATION LOOP
# ==========================================
def main():
    print("=" * 65)
    print("🚀 INITIALIZING SYNAPSE-X MULTI-AGENT COMMERCE CORE")
    print("=" * 65)

    mia = MarketIntelligenceAgent()
    pba = PersonaBehaviorAgent()
    soa = StrategyPolicyAgent()
    mca = MetaCriticAgent()

    policy_bias = 1.00
    test_cycles = [
        {"intent": 0.85, "churn": 0.20, "surge": 1.4, "inventory": 80, "converted": True},
        {"intent": 0.45, "churn": 0.75, "surge": 0.9, "inventory": 15, "converted": False},
        {"intent": 0.92, "churn": 0.10, "surge": 1.8, "inventory": 45, "converted": True},
    ]

    for idx, cycle in enumerate(test_cycles, 1):
        print(f"\n--- [CYCLE {idx}] Ingesting Live Telemetry ---")
        telemetry = SessionTelemetry(
            user_id=f"USR-890{idx}",
            clickstream_intent_score=cycle["intent"],
            churn_probability=cycle["churn"],
            current_cart_value=120.0
        )
        market = MarketContext(
            sku_id="SKU-NEO-X",
            competitor_avg_price=249.99,
            demand_surge_factor=cycle["surge"],
            inventory_count=cycle["inventory"]
        )

        # Agent Reasoning
        m_analysis = mia.analyze(market)
        p_analysis = pba.evaluate(telemetry)
        decision = soa.arbitrate(market, telemetry, m_analysis, p_analysis, policy_bias)
        
        # Self-Evolution Loop
        reward, policy_bias = mca.evaluate_and_evolve(cycle["converted"], policy_bias)

        print(f"📊 Market Insight   : Ceiling = ${m_analysis['recommended_base_ceiling']:.2f} | Risk = {m_analysis['inventory_risk']}")
        print(f"👤 Persona Insight  : Affinity Score = {p_analysis['composite_affinity']} | Discount = {p_analysis['max_allowable_discount_pct']}%")
        print(f"⚡ Strategy Decision: Price = ${decision.adjusted_price} | Bundles = {decision.recommended_bundle}")
        print(f"🔄 Meta-Critic Loop : Reward = {reward} | Updated Policy Bias = {policy_bias}")

if __name__ == "__main__":
    main()