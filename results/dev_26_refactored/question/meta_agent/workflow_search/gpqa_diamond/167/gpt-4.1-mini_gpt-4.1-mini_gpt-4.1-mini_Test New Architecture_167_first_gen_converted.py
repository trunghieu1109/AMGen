async def forward_167(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize the given information about the four issues and the answer choices, "
        "clarifying their definitions and relevance in genomics data analysis."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1,
        n_repeat=self.max_sc
    )
    logs.append(log_0_1)

    cot_instruction_1_1 = (
        "Sub-task 1: Analyze and integrate the relationships and impacts of the four issues on genomics data analysis, "
        "focusing on how each can cause difficult-to-spot errors."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent insights about the relationships and impacts of the four issues."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_reflect_instruction_1_2 = (
        "Sub-task 2: Contextualize the issues within the field of genomics data analysis, "
        "considering common scenarios and data workflows where these errors arise."
    )
    critic_instruction_1_2 = (
        "Please review and provide the limitations of provided solutions regarding the contextualization of these issues in genomics data analysis workflows."
    )
    cot_reflect_desc_1_2 = {
        "instruction": cot_reflect_instruction_1_2,
        "critic_instruction": critic_instruction_1_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Sub-task 1: Evaluate the four issues against the criteria of being the most common and difficult-to-spot sources of erroneous results, "
        "and select the correct combination from the given choices."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Select the best answer choice that correctly identifies the most common and difficult-to-spot erroneous issues in genomics data analysis."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer']],
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
