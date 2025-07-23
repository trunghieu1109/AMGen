async def forward_180(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": []}

    for iteration in range(3):
        cot_instruction_0 = (
            "Sub-task 0: Extract and summarize all given information from the query relevant to solar neutrino fluxes and the hypothetical stopping of the pp-III branch."
        )
        cot_agent_desc_0 = {
            "instruction": cot_instruction_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0, log_0 = await self.cot(
            subtask_id=f"stage_0_subtask_0_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0
        )
        stage0_results["subtask_0"].append(results_0)
        logs.append(log_0)

        cot_instruction_1 = (
            "Sub-task 1: Analyze relationships between solar neutrino production branches, their energy spectra, and how stopping pp-III affects neutrino flux in the two specified energy bands."
        )
        cot_agent_desc_1 = {
            "instruction": cot_instruction_1,
            "input": [taskInfo, results_0["thinking"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0"]
        }
        results_1, log_1 = await self.cot(
            subtask_id=f"stage_0_subtask_1_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_1
        )
        stage0_results["subtask_1"].append(results_1)
        logs.append(log_1)

        cot_instruction_2 = (
            "Sub-task 2: Identify and clarify any assumptions or missing data needed to estimate the flux ratio, including ignoring flavor oscillations and considering neutrino travel time."
        )
        cot_agent_desc_2 = {
            "instruction": cot_instruction_2,
            "input": [taskInfo, results_1["thinking"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1"]
        }
        results_2, log_2 = await self.cot(
            subtask_id=f"stage_0_subtask_2_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_2
        )
        stage0_results["subtask_2"].append(results_2)
        logs.append(log_2)

    reflexion_instruction_0 = (
        "Sub-task 0: Consolidate and simplify the intermediate analysis to focus on the key factors determining the flux ratio between the two energy bands."
    )
    reflexion_0_input = [taskInfo] + [res["thinking"] for res in stage0_results["subtask_2"]]
    reflexion_desc_0 = {
        "instruction": reflexion_instruction_0,
        "input": reflexion_0_input,
        "critic_instruction": "Please review and provide the limitations of provided solutions of consolidation and simplification.",
        "temperature": 0.0,
        "context": ["user query"] + [f"thinking of stage_0_subtask_2_iter_{i}" for i in range(3)]
    }
    results_reflexion_0, log_reflexion_0 = await self.reflexion(
        subtask_id="stage_1_subtask_0",
        reflect_desc=reflexion_desc_0,
        n_repeat=self.max_round
    )
    logs.append(log_reflexion_0)

    debate_instruction_1 = (
        "Sub-task 1: Evaluate candidate flux ratio values against the analysis and select the best approximate ratio from the given choices."
    )
    debate_desc_1 = {
        "instruction": debate_instruction_1,
        "final_decision_instruction": "Sub-task 1: Select the best approximate flux ratio from the given choices based on the analysis.",
        "input": [taskInfo, results_reflexion_0["thinking"], results_reflexion_0["answer"]],
        "context": ["user query", "thinking of stage_1_subtask_0", "answer of stage_1_subtask_0"],
        "temperature": 0.5
    }
    results_debate_1, log_debate_1 = await self.debate(
        subtask_id="stage_1_subtask_1",
        debate_desc=debate_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log_debate_1)

    cot_instruction_0_stage2 = (
        "Sub-task 0: Apply the reasoning and assumptions to calculate or estimate the numerical flux ratio Flux(band 1) / Flux(band 2) after stopping the pp-III branch."
    )
    cot_agent_desc_0_stage2 = {
        "instruction": cot_instruction_0_stage2,
        "input": [taskInfo, results_debate_1["thinking"], results_debate_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1_subtask_1", "answer of stage_1_subtask_1"]
    }
    results_0_stage2, log_0_stage2 = await self.cot(
        subtask_id="stage_2_subtask_0",
        cot_agent_desc=cot_agent_desc_0_stage2
    )
    logs.append(log_0_stage2)

    cot_agent_instruction_1_stage2 = (
        "Sub-task 1: Generate the final answer choice based on the calculated flux ratio."
    )
    cot_agent_desc_1_stage2 = {
        "instruction": cot_agent_instruction_1_stage2,
        "input": [taskInfo, results_0_stage2["thinking"], results_0_stage2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2_subtask_0", "answer of stage_2_subtask_0"]
    }
    results_1_stage2, log_1_stage2 = await self.answer_generate(
        subtask_id="stage_2_subtask_1",
        cot_agent_desc=cot_agent_desc_1_stage2
    )
    logs.append(log_1_stage2)

    review_instruction_0_stage3 = (
        "Sub-task 0: Review the final answer and reasoning for correctness, consistency, and alignment with the problem statement and assumptions."
    )
    review_desc_0_stage3 = {
        "instruction": review_instruction_0_stage3,
        "input": [taskInfo, results_1_stage2["thinking"], results_1_stage2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2_subtask_1", "answer of stage_2_subtask_1"]
    }
    results_0_stage3, log_0_stage3 = await self.review(
        subtask_id="stage_3_subtask_0",
        review_desc=review_desc_0_stage3
    )
    logs.append(log_0_stage3)

    final_answer = await self.make_final_answer(results_1_stage2["thinking"], results_1_stage2["answer"])

    return final_answer, logs
