async def forward_12(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Identify and extract the complex expression components and constraints on z from the input expression: "
        "Find the largest possible real part of (75+117i)z + (96+144i)/z where |z|=4. Input content: [taskInfo]"
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
        "Sub-task 2: Categorize the elements into coefficients, variable properties, and target quantity (real part) "
        "based on results from stage_0.subtask_1. Input content: [taskInfo, results_0_1['thinking'], results_0_1['answer']]"
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

    cot_instruction_1_1 = (
        "Sub-task 1: Express the complex expression in terms of magnitude and argument of z using polar form "
        "based on results from stage_0.subtask_2. Input content: [taskInfo, results_0_2['thinking'], results_0_2['answer']]"
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Analyze how the real part depends on the argument of z and simplify the expression accordingly "
        "based on results from stage_1.subtask_1. Input content: [taskInfo, results_1_1['thinking'], results_1_1['answer']]"
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_2_1 = (
        "Sub-task 1: Determine the argument of z that maximizes the real part and calculate the maximum value "
        "based on results from stage_1.subtask_2. Input content: [taskInfo, results_1_2['thinking'], results_1_2['answer']]"
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
