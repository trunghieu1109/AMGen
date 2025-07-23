async def forward_164(self, taskInfo):
    logs = []

    results_stage_0 = {}
    results_stage_1 = {}
    results_stage_2 = {}
    results_stage_3 = {"thinking": [], "answer": []}
    results_stage_4 = {}
    results_stage_5 = {}
    results_stage_6 = {}

    # Sequential control flow: stages 0 to 2

    # stage_0.subtask_1: Extract and categorize relevant information from the query
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and extract key elements, catalyst types, activators, and statements from the query. "
        "Input content: taskInfo containing the question and four statements."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    results_stage_0["stage_0.subtask_1"] = results_0_1
    logs.append(log_0_1)

    # stage_1.subtask_1: Evaluate the correctness of each provided statement using SC_CoT
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Assess each of the four statements for chemical and industrial validity based on extracted information from stage_0.subtask_1. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the correctness evaluation of the four statements."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.sc_cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_sc_desc_1_1, n_repeat=self.max_sc)
    results_stage_1["stage_1.subtask_1"] = results_1_1
    logs.append(log_1_1)

    # stage_2.subtask_1: Analyze relationships and dependencies among catalysts, activators, and polymerization steps
    cot_instruction_2_1 = (
        "Sub-task 1: Determine functional associations and constraints between catalyst systems and activators for branching polymer formation. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1 and stage_1.subtask_1."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"], results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_agent_desc_2_1)
    results_stage_2["stage_2.subtask_1"] = results_2_1
    logs.append(log_2_1)

    # Loop control flow: stage 3 (2 iterations)
    for iteration in range(2):
        inputs_for_3_1 = [taskInfo]
        if iteration > 0:
            inputs_for_3_1 += results_stage_3["answer"] + results_stage_3["thinking"]
        cot_instruction_3_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Produce structured intermediate reasoning and provisional conclusions about the correct statement. "
            "Input content: taskInfo, thinking and answer from stage_2.subtask_1, and all previous iteration answers and thinkings of this subtask."
        )
        cot_agent_desc_3_1 = {
            "instruction": cot_instruction_3_1,
            "input": inputs_for_3_1 + [results_2_1["thinking"], results_2_1["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "previous iteration answers and thinkings of stage_3.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
        }
        results_3_1, log_3_1 = await self.cot(subtask_id="stage_3.subtask_1", cot_agent_desc=cot_agent_desc_3_1)
        results_stage_3["thinking"].append(results_3_1["thinking"])
        results_stage_3["answer"].append(results_3_1["answer"])
        logs.append(log_3_1)

    # Sequential control flow: stages 4 to 6

    # stage_4.subtask_1: Select the best candidate statement based on evaluation and reasoning
    cot_instruction_4_1 = (
        "Sub-task 1: Evaluate candidate statements against criteria and select the most accurate and consistent one. "
        "Input content: taskInfo, all thinking and answer results from stage_3.subtask_1 iterations."
    )
    cot_agent_desc_4_1 = {
        "instruction": cot_instruction_4_1,
        "input": [taskInfo] + results_stage_3["thinking"] + results_stage_3["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of all iterations of stage_3.subtask_1", "answer of all iterations of stage_3.subtask_1"]
    }
    results_4_1, log_4_1 = await self.cot(subtask_id="stage_4.subtask_1", cot_agent_desc=cot_agent_desc_4_1)
    results_stage_4["stage_4.subtask_1"] = results_4_1
    logs.append(log_4_1)

    # stage_5.subtask_1: Assess the validity of the selected statement using Debate
    debate_instruction_5_1 = (
        "Sub-task 1: Assess the validity of the selected statement from stage_4.subtask_1 by debating its correctness and consistency. "
        "Input content: taskInfo, thinking and answer from stage_4.subtask_1."
    )
    final_decision_instruction_5_1 = (
        "Sub-task 1: Provide a final decision on the validity of the selected statement based on the debate."
    )
    debate_desc_5_1 = {
        "instruction": debate_instruction_5_1,
        "final_decision_instruction": final_decision_instruction_5_1,
        "input": [taskInfo, results_4_1["thinking"], results_4_1["answer"]],
        "context_desc": ["user query", "thinking of stage_4.subtask_1", "answer of stage_4.subtask_1"],
        "temperature": 0.5
    }
    results_5_1, log_5_1 = await self.debate(subtask_id="stage_5.subtask_1", debate_desc=debate_desc_5_1, n_repeat=self.max_round)
    results_stage_5["stage_5.subtask_1"] = results_5_1
    logs.append(log_5_1)

    # stage_6.subtask_1: Consolidate and refine the final answer
    review_instruction_6_1 = (
        "Sub-task 1: Simplify and consolidate evaluation results from stage_4.subtask_1 and stage_5.subtask_1 to produce a clear final answer. "
        "Input content: taskInfo, thinking and answer from stage_4.subtask_1 and stage_5.subtask_1."
    )
    review_desc_6_1 = {
        "instruction": review_instruction_6_1,
        "input": [taskInfo, results_4_1["thinking"], results_4_1["answer"], results_5_1["thinking"], results_5_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_4.subtask_1", "answer of stage_4.subtask_1", "thinking of stage_5.subtask_1", "answer of stage_5.subtask_1"]
    }
    results_6_1, log_6_1 = await self.review(subtask_id="stage_6.subtask_1", review_desc=review_desc_6_1)
    results_stage_6["stage_6.subtask_1"] = results_6_1
    logs.append(log_6_1)

    final_answer = await self.make_final_answer(results_6_1["thinking"], results_6_1["answer"])
    return final_answer, logs
