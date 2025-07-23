async def forward_23(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Define variables for each digit in the 2x3 grid and introduce explicit carry variables for the row sum addition to correctly model all carry-over cases. "
        "Formulate the row sum equation (two 3-digit numbers summing to 999) including carry variables u0 and u1, as well as digit remainder variables r0 and r1, to avoid the incorrect assumption of zero carry. "
        "This subtask addresses the previous failure of ignoring carry-overs in the row sum. Input content: taskInfo"
    )
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Formulate the column sum constraint (sum of three 2-digit numbers equals 99) as an equation linking the digits, considering that leading zeros are allowed. "
        "Also, clarify the interpretation of numbers formed by rows and columns, ensuring consistent treatment of leading zeros as per the example. "
        "This subtask depends on the digit variables defined in stage_0.subtask_1. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1"
    )
    cot_agent_desc_0_2 = {
        'instruction': cot_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Derive the full system of equations and inequalities linking digits and carry variables from both row and column sum constraints, "
        "ensuring all constraints are consistent and no invalid assumptions (such as zero carry) are made. This includes bounds on digits (0-9) and carries (0 or 1). "
        "This subtask integrates results from stage_0.subtask_1 and stage_0.subtask_2. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1 and stage_0.subtask_2"
    )
    cot_agent_desc_0_3 = {
        'instruction': cot_instruction_0_3,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_0_3, log_0_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_0_3)
    logs.append(log_0_3)

    cot_instruction_1_1 = (
        "Sub-task 1: Systematically enumerate all possible digit assignments (0-9) and carry values (0 or 1) for the grid cells and carry variables that satisfy the row sum constraint with explicit carries as formulated in stage_0.subtask_1 and stage_0.subtask_3. "
        "This enumeration must avoid the previous error of ignoring carry cases and ensure completeness. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_3"
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo, results_0_3['thinking'], results_0_3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Filter the enumerated candidate assignments from stage_1.subtask_1 to retain only those that also satisfy the column sum constraint (sum of three 2-digit numbers equals 99) as formulated in stage_0.subtask_2 and integrated in stage_0.subtask_3. "
        "This ensures that only fully valid candidates remain. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_1, stage_0.subtask_2, and stage_0.subtask_3"
    )
    cot_agent_desc_1_2 = {
        'instruction': cot_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_0_2['thinking'], results_0_2['answer'], results_0_3['thinking'], results_0_3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
    }
    results_1_2, log_1_2 = await self.cot(subtask_id='stage_1.subtask_2', cot_agent_desc=cot_agent_desc_1_2)
    logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Sub-task 1: Perform a thorough verification of each candidate digit assignment from stage_1.subtask_2 against all problem constraints, including digit ranges, carry consistency, and sum conditions, to ensure no invalid solutions are counted. "
        "Count the total number of valid solutions. This subtask explicitly addresses the previous undercounting due to ignoring carry cases and uses a Debate agent collaboration to leverage multiple perspectives for robust validation. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_2"
    )
    final_decision_instruction_2_1 = "Sub-task 1: Verify and count valid digit assignments satisfying all constraints."
    debate_desc_2_1 = {
        'instruction': debate_instruction_2_1,
        'final_decision_instruction': final_decision_instruction_2_1,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'context': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2'],
        'temperature': 0.5
    }
    results_2_1, log_2_1 = await self.debate(subtask_id='stage_2.subtask_1', debate_desc=debate_desc_2_1, n_repeat=self.max_round)
    logs.append(log_2_1)

    cot_reflect_instruction_3_1 = (
        "Sub-task 1: Simplify, consolidate, and format the count of valid digit assignments obtained from stage_2.subtask_1 to produce the final answer. "
        "Ensure clarity and correctness in the presentation of results. "
        "Input content: taskInfo, thinking and answer from stage_2.subtask_1"
    )
    critic_instruction_3_1 = "Please review and provide the limitations of provided solutions of stage_2.subtask_1."
    cot_reflect_desc_3_1 = {
        'instruction': cot_reflect_instruction_3_1,
        'critic_instruction': critic_instruction_3_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_3_1, log_3_1 = await self.reflexion(subtask_id='stage_3.subtask_1', reflect_desc=cot_reflect_desc_3_1, n_repeat=self.max_round)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
