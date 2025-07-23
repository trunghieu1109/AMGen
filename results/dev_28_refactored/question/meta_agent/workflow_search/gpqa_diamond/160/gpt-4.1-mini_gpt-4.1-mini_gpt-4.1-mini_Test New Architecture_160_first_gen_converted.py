async def forward_160(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given information from the query, including system parameters, vacuum conditions, initial mean free path λ1, and observed change to λ2 upon electron beam initiation."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Analyze the physical relationships between mean free path, vacuum parameters, and electron scattering effects to understand why λ2 differs from λ1 despite constant temperature."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent explanation for the change from λ1 to λ2 based on physical principles."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Classify and validate the entities λ1 and λ2, interpret the significance of the factor 1.22, and analyze the relationship between λ1 and λ2 in the context of electron-gas molecule interactions."
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Select the best candidate conclusion about the relationship between λ2 and λ1 from the given choices, based on the physical analysis and classification."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Choose the most physically consistent conclusion regarding λ2 relative to λ1."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
