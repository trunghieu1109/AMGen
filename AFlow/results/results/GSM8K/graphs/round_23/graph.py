from typing import Literal
import metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.Gsm8K.graphs.round_23.prompt as prompt_custom
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
        
        # Double-check the total cost calculation
        total_cost_check = await self.custom(input=problem + f"\nPrevious solution: {final_solution['response']}", instruction=prompt_custom.TOTAL_COST_CHECK_PROMPT)
        
        if verification['output'] and total_cost_check['response']:
            return total_cost_check['response'], self.llm.cost_manager.total_cost
        else:
            return final_solution['response'], self.llm.cost_manager.total_cost
                    