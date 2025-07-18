async def forward_152(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Analyze the first Michael addition reaction (dimethyl malonate + methyl (E)-3-(p-tolyl)acrylate + NaOEt/EtOH) "
        "to determine the expected product structure and key features with context from the query."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the first Michael addition reaction product."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze the second Michael addition reaction (1-(cyclohex-1-en-1-yl)piperidine + (E)-but-2-enenitrile + MeOH/H3O+) "
        "to determine the expected product structure and key features with context from the query and output of subtask 1."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the second Michael addition reaction product."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Analyze the third Michael addition reaction (compound C + but-3-en-2-one + KOH/H2O) "
        "to infer the identity of compound C and the expected product structure with context from the query and outputs of subtasks 1 and 2."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent answer for the third Michael addition reaction product and compound C identity."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Integrate the analyses from the three Michael addition reactions to assign the correct product names and structures, "
        "considering tautomerism, substitution patterns, and resonance forms, using outputs from subtasks 1, 2, and 3."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of the provided solutions for integrating the three Michael addition reaction analyses."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'critic_instruction': critic_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Evaluate the four multiple-choice options against the integrated product assignments from subtask 4 "
        "to select the correct set of products A, B, and C."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Select the correct multiple-choice answer for the Michael addition products A, B, and C based on integrated analysis."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'context_desc': ["user query", "thinking of subtask 4", "answer of subtask 4"],
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
