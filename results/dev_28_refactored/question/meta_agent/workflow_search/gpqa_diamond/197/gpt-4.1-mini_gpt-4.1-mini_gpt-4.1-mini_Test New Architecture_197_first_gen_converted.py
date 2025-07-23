async def forward_197(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and organize all given information into structured intermediate representations, "
        "including defining species, their stoichiometry, and equilibrium expressions using the cumulative stability constants. "
        "Use the provided cobalt and SCN- concentrations and stability constants to define the species Co2+, Co(SCN)+, Co(SCN)2, Co(SCN)3-, and Co(SCN)4^2-."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, derive the expressions for the concentrations of all cobalt species "
        "(free Co(II), mono-, di-, tri-, and tetrathiocyanato complexes) using the cumulative stability constants and given concentrations, "
        "and calculate their numerical values. Assume free SCN- concentration remains constant at 0.1 M."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent numerical concentrations for all cobalt species based on the calculations."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Calculate the percentage of the blue dithiocyanato cobalt(II) complex (Co(SCN)2) relative to the total cobalt concentration "
        "using the concentrations derived in Sub-task 2. Provide reasoning and verify the calculation correctness."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of the provided solutions for the percentage calculation of the dithiocyanato complex."
    )
    cot_reflect_desc3 = {
        "instruction": cot_reflect_instruction3,
        "critic_instruction": critic_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    formatter_instruction4 = (
        "Sub-task 4: Format and summarize the final result clearly, comparing it with the provided choices and identifying the correct answer. "
        "Provide a concise final answer statement."
    )
    formatter_desc4 = {
        "instruction": formatter_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "format": "short and concise, without explanation"
    }
    results4, log4 = await self.specific_format(
        subtask_id="subtask_4",
        formatter_desc=formatter_desc4
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
