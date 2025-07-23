async def forward_159(self, taskInfo):
    logs = []
    stage_0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": []}

    for iteration in range(3):
        cot_instruction_0_0 = (
            "Sub-task 0: Extract and summarize all given information from the query, including aperture shape, light properties, and problem conditions."
        )
        cot_agent_desc_0_0 = {
            "instruction": cot_instruction_0_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id=f"stage_0_subtask_0_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)
        stage_0_results["subtask_0"].append(results_0_0)

        cot_instruction_0_1 = (
            "Sub-task 1: Analyze the relationship between the polygonal aperture and the limiting circular aperture as N approaches infinity, including the role of the apothem."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo, results_0_0["thinking"], results_0_0["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id=f"stage_0_subtask_1_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        stage_0_results["subtask_1"].append(results_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Identify the relevant diffraction theory for a circular aperture and the mathematical form of minima positions (Bessel function zeros)."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id=f"stage_0_subtask_2_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        stage_0_results["subtask_2"].append(results_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Apply the small-angle approximation to simplify the angular position expressions of diffraction minima."
        )
        cot_agent_desc_0_3 = {
            "instruction": cot_instruction_0_3,
            "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id=f"stage_0_subtask_3_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_3
        )
        logs.append(log_0_3)
        stage_0_results["subtask_3"].append(results_0_3)

    last_subtask_3 = stage_0_results["subtask_3"][-1]

    cot_reflect_instruction_1_0 = (
        "Sub-task 0: Consolidate and simplify the expressions for the angular positions of the first two minima based on the previous analysis."
    )
    critic_instruction_1_0 = (
        "Please review and provide the limitations of provided solutions of consolidation and simplification of angular positions of minima."
    )
    cot_reflect_desc_1_0 = {
        "instruction": cot_reflect_instruction_1_0,
        "critic_instruction": critic_instruction_1_0,
        "input": [taskInfo, last_subtask_3["thinking"], last_subtask_3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_1_0, log_1_0 = await self.reflexion(
        subtask_id="stage_1_subtask_0",
        reflect_desc=cot_reflect_desc_1_0,
        n_repeat=self.max_round
    )
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = (
        "Sub-task 1: Calculate the angular distance between the first two minima using the simplified expressions."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_agent_instruction_1_1,
        "input": [taskInfo, results_1_0["thinking"], results_1_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_0", "answer of stage_1.subtask_0"]
    }
    results_1_1, log_1_1 = await self.answer_generate(
        subtask_id="stage_1_subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    aggregate_instruction_1_2 = (
        "Sub-task 2: Compare the calculated angular distance with the given choices and select the best matching candidate."
    )
    aggregate_desc_1_2 = {
        "instruction": aggregate_instruction_1_2,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "solutions generated from stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.aggregate(
        subtask_id="stage_1_subtask_2",
        aggregate_desc=aggregate_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_2_0 = (
        "Sub-task 0: Apply the transformation from polygonal aperture diffraction to circular aperture diffraction in the limit N → ∞, confirming the radius equals the apothem."
    )
    cot_agent_desc_2_0 = {
        "instruction": cot_instruction_2_0,
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_0, log_2_0 = await self.cot(
        subtask_id="stage_2_subtask_0",
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    cot_agent_instruction_2_0_answer = (
        "Sub-task 0: Generate final answer confirming the transformation and radius-apothem equivalence."
    )
    cot_agent_desc_2_0_answer = {
        "instruction": cot_agent_instruction_2_0_answer,
        "input": [taskInfo, results_2_0["thinking"], results_2_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_0", "answer of stage_2.subtask_0"]
    }
    results_2_0_answer, log_2_0_answer = await self.answer_generate(
        subtask_id="stage_2_subtask_0_answer",
        cot_agent_desc=cot_agent_desc_2_0_answer
    )
    logs.append(log_2_0_answer)

    review_instruction_3_0 = (
        "Sub-task 0: Validate the final selected angular distance against known physical optics results and problem assumptions (small angle, aperture shape)."
    )
    review_desc_3_0 = {
        "instruction": review_instruction_3_0,
        "input": [taskInfo, results_2_0_answer["thinking"], results_2_0_answer["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_0_answer", "answer of stage_2.subtask_0_answer"]
    }
    results_3_0, log_3_0 = await self.review(
        subtask_id="stage_3_subtask_0",
        review_desc=review_desc_3_0
    )
    logs.append(log_3_0)

    final_answer = await self.make_final_answer(results_3_0["thinking"], results_3_0["answer"])

    return final_answer, logs
