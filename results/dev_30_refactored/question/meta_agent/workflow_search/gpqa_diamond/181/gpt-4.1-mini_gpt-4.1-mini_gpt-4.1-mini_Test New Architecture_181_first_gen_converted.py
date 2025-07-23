async def forward_181(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize the given Mott-Gurney equation and its parameters from the query. "
        "Input content: taskInfo"
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
        "Sub-task 2: Analyze the physical meaning and assumptions underlying the Mott-Gurney equation, including SCLC regime and device characteristics. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1"
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
        "Sub-task 3: Interpret the conditions described in the four candidate statements regarding device type, contact type, and current components. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1"
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_instruction_0_4 = (
        "Sub-task 4: Identify key criteria that determine the validity of the Mott-Gurney equation based on the summarized information and physical principles. "
        "Input content: thinking and answer from stage_0.subtask_2 and stage_0.subtask_3"
    )
    cot_agent_desc_0_4 = {
        "instruction": cot_instruction_0_4,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_0_3['thinking'], results_0_3['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_0_4, log_0_4 = await self.cot(
        subtask_id="stage_0.subtask_4",
        cot_agent_desc=cot_agent_desc_0_4
    )
    logs.append(log_0_4)

    aggregate_instruction_1_1 = (
        "Sub-task 1: Combine the extracted equation details and physical assumptions into a consolidated summary of validity conditions. "
        "Input content: thinking and answer from stage_0.subtask_1, stage_0.subtask_2, stage_0.subtask_3, and stage_0.subtask_4"
    )
    aggregate_desc_1_1 = {
        "instruction": aggregate_instruction_1_1,
        "input": [taskInfo, 
                  results_0_1['thinking'], results_0_1['answer'],
                  results_0_2['thinking'], results_0_2['answer'],
                  results_0_3['thinking'], results_0_3['answer'],
                  results_0_4['thinking'], results_0_4['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking and answers of stage_0 subtasks"]
    }
    results_1_1, log_1_1 = await self.aggregate(
        subtask_id="stage_1.subtask_1",
        aggregate_desc=aggregate_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Evaluate the consistency of each candidate statement with the consolidated validity conditions. "
        "Input content: thinking and answer from stage_1.subtask_1"
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

    debate_instruction_2_1 = (
        "Sub-task 1: Validate each candidate statement against the consolidated criteria to determine which statements are true or false. "
        "Input content: thinking and answer from stage_1.subtask_2"
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Validate each candidate statement against the consolidated criteria and decide which are true or false."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=2
    )
    logs.append(log_2_1)

    review_instruction_2_2 = (
        "Sub-task 2: Select the statement(s) that satisfy all validity conditions of the Mott-Gurney equation. "
        "Input content: thinking and answer from stage_2.subtask_1"
    )
    review_desc_2_2 = {
        "instruction": review_instruction_2_2,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.review(
        subtask_id="stage_2.subtask_2",
        review_desc=review_desc_2_2
    )
    logs.append(log_2_2)

    cot_instruction_2_3 = (
        "Sub-task 3: Assess the overall validity and produce a comprehensive evaluation outcome identifying the correct statement. "
        "Input content: thinking and answer from stage_2.subtask_2"
    )
    cot_agent_desc_2_3 = {
        "instruction": cot_instruction_2_3,
        "input": [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_2_3, log_2_3 = await self.cot(
        subtask_id="stage_2.subtask_3",
        cot_agent_desc=cot_agent_desc_2_3
    )
    logs.append(log_2_3)

    formatter_instruction_3_1 = (
        "Sub-task 1: Format the evaluation outcome into a clear, concise final answer indicating which statement is true about the Mott-Gurney equation validity. "
        "Input content: thinking and answer from stage_2.subtask_3"
    )
    formatter_desc_3_1 = {
        "instruction": formatter_instruction_3_1,
        "input": [taskInfo, results_2_3['thinking'], results_2_3['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_3", "answer of stage_2.subtask_3"],
        "format": "short and concise, without explanation"
    }
    results_3_1, log_3_1 = await self.specific_format(
        subtask_id="stage_3.subtask_1",
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
