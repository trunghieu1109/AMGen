from typing import Literal
import metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.round_16.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the graph
        """
        solutions = []
        for _ in range(5):  # Generate 5 solutions
            solution = await self.custom(input=problem, instruction=prompt_custom.MATH_SOLVE_PROMPT)
            solutions.append(solution['response'])
        
        final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        # Add a verification step using the Programmer operator
        verification = await self.programmer(problem=problem, analysis=final_solution['response'])
        
        # Double-check the final answer with a focus on total cost calculation
        double_check = await self.custom(input=problem + f"\nPrevious solution: {verification['output']}", instruction=prompt_custom.DOUBLE_CHECK_PROMPT)
        
        if double_check['response'] != verification['output']:
            return double_check['response'], self.llm.cost_manager.total_cost
        else:
            return verification['output'], self.llm.cost_manager.total_cost
                    