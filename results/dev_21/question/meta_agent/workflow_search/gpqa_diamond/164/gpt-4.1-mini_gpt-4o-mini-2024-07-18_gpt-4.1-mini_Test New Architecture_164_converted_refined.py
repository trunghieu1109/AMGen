async def forward_164(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the catalyst systems involved in ethylene polymerization, "
        "including the initial homogeneous organometallic catalyst producing high-density polymer and the second catalyst system intended to introduce regular branches using only ethylene. "
        "Explicitly identify the chemical nature and mechanistic roles of these catalysts. Avoid uncritical acceptance of statements; critically evaluate catalyst claims based on known polymer chemistry."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Analyze and classify the activators mentioned, focusing on aluminum-based activators and specific activators compatible with group VIa transition metal catalysts. "
        "Critically evaluate their roles and effectiveness in the essential additional reaction step, explicitly addressing the incompatibility of aluminum-based activators and implications for catalyst performance. Avoid superficial plausibility and require chemical evidence."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Analyze and classify the industrial and economic context of the catalyst systems, including the claim of combined dual catalyst systems implemented industrially in the US and the cost implications of noble metal catalysts. "
        "Critically verify the industrial reality of such systems by cross-referencing known commercial practices, patents, or literature. Explicitly challenge plausibility and identify contradictions or gaps in industrial feasibility and economic viability."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Verify the industrial implementation claim and economic feasibility of the dual catalyst system for producing branched polyethylene using only ethylene. "
        "Integrate authoritative external knowledge or references to confirm or refute the existence of such systems at industrial scale in the US, and assess the practical viability of group VIa catalysts with specific activators versus noble metal catalysts. "
        "This step addresses the previous failure to cross-validate industrial claims and economic context."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Perform a structured evidence synthesis and final evaluation of the correctness of each of the four statements regarding the formation of branched polymers using only ethylene and a dual catalyst system. "
        "Explicitly incorporate and reconcile the outputs and confidence levels from subtasks 1, 2, 3, and 4, comparing each statement against chemical, industrial, and economic evidence. "
        "Avoid ignoring contradictory subtask results or relying on superficial plausibility. Select the uniquely supported statement based on rigorous cross-validation and critical analysis."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
