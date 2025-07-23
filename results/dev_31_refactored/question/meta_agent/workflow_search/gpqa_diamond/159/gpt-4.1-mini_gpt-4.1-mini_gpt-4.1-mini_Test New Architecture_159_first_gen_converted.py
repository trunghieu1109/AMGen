async def forward_159(self, taskInfo):
    logs = []

    results = {}

    stage_0_results = {}

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the polygon aperture as N approaches infinity and approximate it as a circular aperture of radius a. "
        "Input content: taskInfo (user query)"
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)
    stage_0_results["stage_0.subtask_1"] = results_0_1

    cot_instruction_0_2 = (
        "Sub-task 2: Derive the angular positions of the first two minima in the far-field diffraction pattern using the Airy pattern formula and small angle approximation. "
        "Input content: results (thinking and answer) from stage_0.subtask_1"
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)
    stage_0_results["stage_0.subtask_2"] = results_0_2

    cot_instruction_0_3 = (
        "Sub-task 3: Calculate the angular distance between the first two minima and express it in terms of lambda and a. "
        "Input content: results (thinking and answer) from stage_0.subtask_2"
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)
    stage_0_results["stage_0.subtask_3"] = results_0_3

    stage_1_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_sc_instruction_1_1 = (
            "Sub-task 1: Compare the computed angular distance with each provided choice to assess closeness. "
            "Input content: results (thinking and answer) from stage_0.subtask_3"
        )
        final_decision_instruction_1_1 = (
            "Sub-task 1: Synthesize and choose the most consistent answer for the comparison of computed angular distance with choices."
        )
        cot_sc_desc_1_1 = {
            "instruction": cot_sc_instruction_1_1,
            "final_decision_instruction": final_decision_instruction_1_1,
            "input": [taskInfo, stage_0_results["stage_0.subtask_3"]["thinking"], stage_0_results["stage_0.subtask_3"]["answer"]] + stage_1_results["stage_1.subtask_1"]["thinking"] + stage_1_results["stage_1.subtask_1"]["answer"],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"] + ["previous thinking of stage_1.subtask_1"] * len(stage_1_results["stage_1.subtask_1"]["thinking"])
        }
        results_1_1, log_1_1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_1_1,
            n_repeat=3
        )
        logs.append(log_1_1)
        stage_1_results["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        stage_1_results["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        aggregate_instruction_1_2 = (
            "Sub-task 2: From solutions generated in Subtask 1, aggregate these solutions and return the consistent and the best solution for selecting the candidate choice that best matches the computed angular distance. "
            "Input content: results (thinking and answer) from stage_1.subtask_1"
        )
        aggregate_desc_1_2 = {
            "instruction": aggregate_instruction_1_2,
            "input": [taskInfo] + stage_1_results["stage_1.subtask_1"]["answer"] + stage_1_results["stage_1.subtask_1"]["thinking"],
            "temperature": 0.0,
            "context": ["user query", "solutions generated from stage_1.subtask_1"]
        }
        results_1_2, log_1_2 = await self.aggregate(
            subtask_id="stage_1.subtask_2",
            aggregate_desc=aggregate_desc_1_2
        )
        logs.append(log_1_2)
        stage_1_results["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        stage_1_results["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    cot_agent_instruction_2_1 = (
        "Sub-task 1: Generate the final answer choice corresponding to the best evaluated candidate. "
        "Input content: results (thinking and answer) from stage_1.subtask_2"
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_agent_instruction_2_1,
        "input": [taskInfo] + stage_1_results["stage_1.subtask_2"]["thinking"] + stage_1_results["stage_1.subtask_2"]["answer"],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.answer_generate(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1["thinking"], results_2_1["answer"])
    return final_answer, logs
