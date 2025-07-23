async def forward_154(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Construct explicit matrix representations of the operators P_x, P_y, and P_z, "
        "and confirm the given state vector is normalized and expressed in the P_z eigenbasis, "
        "with context from the user query."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2_1 = (
        "Sub-task 2.1: Compute the expectation value <P_z> in the given state by applying the P_z operator "
        "to the state vector and taking the inner product, based on the output from stage_1.subtask_1."
    )
    final_decision_instruction2_1 = (
        "Sub-task 2.1: Synthesize and choose the most consistent value for <P_z> expectation value."
    )
    cot_sc_desc2_1 = {
        "instruction": cot_sc_instruction2_1,
        "final_decision_instruction": final_decision_instruction2_1,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2_1, log2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2_1,
        n_repeat=self.max_sc
    )
    logs.append(log2_1)

    cot_sc_instruction2_2 = (
        "Sub-task 2.2: Compute the expectation value <P_z^2> in the given state by applying the P_z^2 operator "
        "(square of P_z matrix) to the state vector and taking the inner product, based on the output from stage_1.subtask_1."
    )
    final_decision_instruction2_2 = (
        "Sub-task 2.2: Synthesize and choose the most consistent value for <P_z^2> expectation value."
    )
    cot_sc_desc2_2 = {
        "instruction": cot_sc_instruction2_2,
        "final_decision_instruction": final_decision_instruction2_2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2_2, log2_2 = await self.sc_cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_sc_desc2_2,
        n_repeat=self.max_sc
    )
    logs.append(log2_2)

    cot_instruction3 = (
        "Sub-task 3: Calculate the uncertainty ΔP_z = sqrt(<P_z^2> - <P_z>^2) using the expectation values "
        "obtained from stage_2.subtask_1 and stage_2.subtask_2."
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "input": [taskInfo, results2_1["thinking"], results2_1["answer"], results2_2["thinking"], results2_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results3, log3 = await self.cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc3
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Compare the calculated uncertainty ΔP_z with the provided choices and select the best matching candidate, "
        "based on the output from stage_3.subtask_1."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the best matching uncertainty ΔP_z from the given choices."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_4.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
