async def forward_18(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and represent the given points O, A, B and segment AB in coordinate form; "
        "define the family F of unit segments PQ with P on x-axis and Q on y-axis in the first quadrant. "
        "Input: taskInfo"
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

    cot_instruction_0_2 = (
        "Sub-task 2: Analyze the geometric and algebraic conditions that characterize segments in F, "
        "including the unit length constraint and first quadrant location. "
        "Input: taskInfo, thinking and answer from stage_0.subtask_1"
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Formulate the condition for a point on segment AB to lie on some segment in F other than AB, "
        "and characterize the uniqueness condition for point C. "
        "Input: taskInfo, thinking and answer from stage_0.subtask_2"
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_instruction_1_1 = (
        "Sub-task 1: Using the conditions from stage_0, determine the unique point C on segment AB that does not lie on any other segment from F except AB. "
        "Input: taskInfo, thinking and answer from stage_0.subtask_3"
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_3['thinking'], results_0_3['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    debate_instruction_2_1 = (
        "Sub-task 1: Compute the squared distance OC^2 for the point C found in stage_1, simplify it to lowest terms p/q, "
        "and find the sum p+q. "
        "Input: taskInfo, thinking and answer from stage_1.subtask_1"
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Provide the simplified fraction p/q for OC^2 and the sum p+q as the final answer."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=2
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
