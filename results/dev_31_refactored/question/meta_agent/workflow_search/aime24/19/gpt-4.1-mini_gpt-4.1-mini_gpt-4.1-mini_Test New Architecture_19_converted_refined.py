async def forward_19(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Express the polynomial inside the product as a function of x and simplify it algebraically, "
        "clarifying the inclusion of the root at 1 despite the problem statement ω ≠ 1. Input content: taskInfo"
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Relate the polynomial evaluated at 13th roots of unity to known cyclotomic polynomials or factorization properties, "
        "using the simplified form from stage_0.subtask_1. Input content: taskInfo, thinking and answer from stage_0.subtask_1"
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: Rewrite the product over all 13th roots of unity in terms of a simpler expression involving powers of (1+i), "
        "based on the factorization from stage_0.subtask_2. Input content: taskInfo, thinking and answer from stage_0.subtask_2"
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Compute (1+i)^4 and (1+i)^8 algebraically to establish (1+i)^{12}, then multiply by (1+i) to get (1+i)^{13}, "
        "carefully tracking all signs and factors to avoid errors like dropping the overall -1 factor. Input content: taskInfo, thinking and answer from stage_1.subtask_1"
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_1_3 = (
        "Sub-task 3: Verify the result of (1+i)^{13} by cross-checking the algebraic computation with the polar form calculation to ensure no sign or phase errors, "
        "preventing the mistake of omitting factors like e^{i3π} = -1. Input content: taskInfo, thinking and answer from stage_1.subtask_2"
    )
    cot_agent_desc_1_3 = {
        "instruction": cot_instruction_1_3,
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_1_3, log_1_3 = await self.cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_agent_desc_1_3
    )
    logs.append(log_1_3)

    cot_instruction_1_4 = (
        "Sub-task 4: Use the verified value of (1+i)^{13} to compute the original product and then reduce the result modulo 1000 to find the remainder, "
        "ensuring all intermediate steps respect modular arithmetic constraints. Input content: taskInfo, thinking and answer from stage_1.subtask_3"
    )
    cot_agent_desc_1_4 = {
        "instruction": cot_instruction_1_4,
        "input": [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results_1_4, log_1_4 = await self.cot(
        subtask_id="stage_1.subtask_4",
        cot_agent_desc=cot_agent_desc_1_4
    )
    logs.append(log_1_4)

    cot_agent_instruction_2_1 = (
        "Sub-task 1: Format the remainder as the final answer and provide a concise summary, "
        "using the modulo 1000 result from stage_1.subtask_4. Input content: taskInfo, thinking and answer from stage_1.subtask_4"
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_agent_instruction_2_1,
        "input": [taskInfo, results_1_4['thinking'], results_1_4['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"]
    }
    results_2_1, log_2_1 = await self.answer_generate(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
