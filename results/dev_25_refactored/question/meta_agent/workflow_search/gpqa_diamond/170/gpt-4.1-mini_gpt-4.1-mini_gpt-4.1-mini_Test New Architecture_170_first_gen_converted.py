async def forward_170(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the given chemical information: identify the six substances, their substituents, "
        "and the reaction conditions (electrophilic bromination with excess bromine, monobromo derivative formation)."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Clarify and define key terms and assumptions, including the meaning of 'weight fraction of the para-isomer' "
        "and the implications of the reaction conditions on isomer formation."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for clarifying key terms and assumptions."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Analyze the electronic and steric effects of each substituent on the benzene ring to determine their directing influence "
        "(ortho/para or meta) and expected para-isomer yield trends."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Debate and finalize the analysis of substituent effects and para-isomer yield trends."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Integrate the substituent effects with steric considerations and reaction conditions to estimate the relative weight fractions "
        "of the para-isomer for each substance."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of provided solutions of estimating para-isomer weight fractions."
    )
    cot_reflect_desc4 = {
        "instruction": cot_reflect_instruction4,
        "critic_instruction": critic_instruction4,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"], results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Select and arrange the six substances in order of increasing weight fraction of the para-isomer yield based on the integrated analysis, "
        "and compare with the given choices to identify the correct sequence."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final ordered sequence of substances by increasing para-isomer weight fraction and identify the matching choice."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "context_desc": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
