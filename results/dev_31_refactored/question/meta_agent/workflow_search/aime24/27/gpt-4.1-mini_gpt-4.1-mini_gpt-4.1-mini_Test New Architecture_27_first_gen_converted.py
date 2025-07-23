async def forward_27(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Represent N as a 4-digit number and formalize the condition that changing any digit to 1 yields a number divisible by 7. "
        "Input content: problem query."
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

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, derive modular arithmetic conditions and relationships among digits from the divisibility constraints. "
        "Input content: problem query, thinking and answer from stage_0.subtask_1."
    )
    final_decision_instruction_0_2 = (
        "Sub-task 2: Synthesize and choose the most consistent modular conditions and digit relationships for the problem."
    )
    cot_sc_desc_0_2 = {
        "instruction": cot_sc_instruction_0_2,
        "final_decision_instruction": final_decision_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc_0_2,
        n_repeat=3
    )
    logs.append(log_0_2)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Generate and test candidate digit combinations satisfying the modular conditions to identify all valid N. "
        "Input content: problem query, thinking and answer from stage_0.subtask_2."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent candidate numbers N satisfying the modular conditions."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=3
    )
    logs.append(log_1_1)

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Select the greatest valid four-digit number N from the candidates. "
        "Input content: problem query, thinking and answer from stage_1.subtask_1."
    )
    final_decision_instruction_1_2 = (
        "Sub-task 2: Synthesize and choose the greatest valid number N from candidates."
    )
    cot_sc_desc_1_2 = {
        "instruction": cot_sc_instruction_1_2,
        "final_decision_instruction": final_decision_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc_1_2,
        n_repeat=3
    )
    logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Sub-task 1: Divide N by 1000 to find quotient Q and remainder R, then compute Q + R. "
        "Input content: problem query, thinking and answer from stage_0.subtask_2 and stage_1.subtask_2."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Synthesize and provide the final answer Q + R from the identified number N."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_1_2['thinking'], results_1_2['answer']],
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
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
