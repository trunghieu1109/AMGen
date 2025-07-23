async def forward_152(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []
    loop_results = {"stage_1.subtask_1": {"thinking": [], "answer": []},
                    "stage_1.subtask_2": {"thinking": [], "answer": []},
                    "stage_1.subtask_3": {"thinking": [], "answer": []}}

    cot_instruction_0 = (
        "Stage 0 - Sub-task 1: Identify and extract key chemical entities, reaction conditions, "
        "and product candidates from the query. Input: taskInfo (query)."
    )
    cot_agent_desc_0 = {
        "instruction": cot_instruction_0,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0, log_0 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0
    )
    logs.append(log_0)

    for iteration in range(2):
        cot_sc_instruction_1 = (
            f"Stage 1 - Sub-task 1 (Iteration {iteration+1}): Analyze Michael addition reaction A "
            "and propose the most plausible product structure. Input: taskInfo, thinking and answer from stage_0.subtask_1."
        )
        final_decision_instruction_1 = (
            "Stage 1 - Sub-task 1 (Iteration {iteration+1}): Synthesize and choose the most consistent answer for reaction A."
        )
        cot_sc_desc_1 = {
            "instruction": cot_sc_instruction_1,
            "final_decision_instruction": final_decision_instruction_1,
            "input": [taskInfo, results_0["thinking"], results_0["answer"]],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_1_1, log_1_1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_1,
            n_repeat=self.max_sc
        )
        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])
        logs.append(log_1_1)

        cot_sc_instruction_2 = (
            f"Stage 1 - Sub-task 2 (Iteration {iteration+1}): Analyze Michael addition reaction B "
            "and propose the most plausible product structure. Input: taskInfo, thinking and answer from stage_0.subtask_1."
        )
        final_decision_instruction_2 = (
            "Stage 1 - Sub-task 2 (Iteration {iteration+1}): Synthesize and choose the most consistent answer for reaction B."
        )
        cot_sc_desc_2 = {
            "instruction": cot_sc_instruction_2,
            "final_decision_instruction": final_decision_instruction_2,
            "input": [taskInfo, results_0["thinking"], results_0["answer"]],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_1_2, log_1_2 = await self.sc_cot(
            subtask_id="stage_1.subtask_2",
            cot_agent_desc=cot_sc_desc_2,
            n_repeat=self.max_sc
        )
        loop_results["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])
        logs.append(log_1_2)

        cot_sc_instruction_3 = (
            f"Stage 1 - Sub-task 3 (Iteration {iteration+1}): Analyze Michael addition reaction C "
            "and propose the most plausible product structure. Input: taskInfo, thinking and answer from stage_0.subtask_1."
        )
        final_decision_instruction_3 = (
            "Stage 1 - Sub-task 3 (Iteration {iteration+1}): Synthesize and choose the most consistent answer for reaction C."
        )
        cot_sc_desc_3 = {
            "instruction": cot_sc_instruction_3,
            "final_decision_instruction": final_decision_instruction_3,
            "input": [taskInfo, results_0["thinking"], results_0["answer"]],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_1_3, log_1_3 = await self.sc_cot(
            subtask_id="stage_1.subtask_3",
            cot_agent_desc=cot_sc_desc_3,
            n_repeat=self.max_sc
        )
        loop_results["stage_1.subtask_3"]["thinking"].append(results_1_3["thinking"])
        loop_results["stage_1.subtask_3"]["answer"].append(results_1_3["answer"])
        logs.append(log_1_3)

    cot_reflect_instruction_4 = (
        "Stage 2 - Sub-task 1: Evaluate all candidate answers against the refined intermediate results "
        "from stage_1.subtask_1, stage_1.subtask_2, and stage_1.subtask_3, and select the best matching choice. "
        "Input: taskInfo, all thinkings and answers from stage_1 subtasks across iterations."
    )
    critic_instruction_4 = (
        "Please review and provide the limitations of provided solutions of stage_1 subtasks, "
        "then synthesize the best final answer choice."
    )
    all_thinkings = []
    all_answers = []
    for subtask_id in ["stage_1.subtask_1", "stage_1.subtask_2", "stage_1.subtask_3"]:
        all_thinkings.extend(loop_results[subtask_id]["thinking"])
        all_answers.extend(loop_results[subtask_id]["answer"])

    cot_reflect_desc_4 = {
        "instruction": cot_reflect_instruction_4,
        "critic_instruction": critic_instruction_4,
        "input": [taskInfo] + all_thinkings + all_answers,
        "temperature": 0.0,
        "context_desc": ["user query"] + all_thinkings + all_answers
    }
    results_2_1, log_2_1 = await self.reflexion(
        subtask_id="stage_2.subtask_1",
        reflect_desc=cot_reflect_desc_4,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1["thinking"], results_2_1["answer"])
    return final_answer, logs
