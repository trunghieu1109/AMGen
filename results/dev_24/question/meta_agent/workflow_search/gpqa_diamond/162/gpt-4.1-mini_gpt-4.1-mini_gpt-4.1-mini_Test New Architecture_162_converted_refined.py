async def forward_162(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Strictly extract and summarize all given quantitative and qualitative information from the query, "
        "including mass of Fe(OH)3, total solution volume, acid concentration, temperature, chemical species involved, "
        "and clarify assumptions about volume definitions (whether acid volume is included in or added to total volume). "
        "Avoid missing key numerical data and ambiguous volume interpretation."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc={
            'instruction': cot_instruction1,
            'final_decision_instruction': "Sub-task 1: Provide a clear, complete summary and assumptions.",
            'input': [taskInfo],
            'context_desc': ["user query"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Analyze the chemical relationships and stoichiometry involved in dissolving Fe(OH)3 in a monobasic strong acid, "
        "including writing balanced chemical equations, identifying the stoichiometric moles of H+ required for complete dissolution, "
        "and clarifying assumptions about acid dissociation and equilibrium. Explicitly incorporate numerical data extracted in Subtask 1."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc={
            'instruction': cot_instruction2,
            'final_decision_instruction': "Sub-task 2: Provide balanced equations and stoichiometric calculations.",
            'input': [taskInfo, results1['answer']],
            'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Calculate the moles of Fe(OH)3 present and the moles of H+ required to fully dissolve it based on stoichiometry and acid concentration. "
        "Then compute the minimum volume of 0.1 M acid needed, explicitly considering whether the acid volume is part of or added to the total solution volume. "
        "Avoid ignoring volume effects and undefined mole values."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent and correct calculations for moles and acid volume."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['answer'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "answer of stage_1.subtask_1", "answer of stage_1.subtask_2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_2.subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4a = (
        "Sub-task 4a: Set up the chemical equilibrium expressions for Fe3+ hydrolysis in the resulting solution, "
        "including the first hydrolysis step (Fe3+ + H2O ⇌ FeOH2+ + H+), using provided or standard hydrolysis constants. "
        "Explicitly incorporate the moles and concentrations from Subtask 3 and consider volume changes due to acid addition. "
        "Avoid neglecting Fe3+ hydrolysis and its effect on pH."
    )
    final_decision_instruction4a = "Sub-task 4a: Provide detailed equilibrium expressions and assumptions."
    debate_desc4a = {
        'instruction': debate_instruction4a,
        'final_decision_instruction': final_decision_instruction4a,
        'input': [taskInfo, results3['answer']],
        'context_desc': ["user query", "answer of stage_2.subtask_3"],
        'temperature': 0.5
    }
    results4a, log4a = await self.debate(
        subtask_id="stage_3.subtask_4a",
        debate_desc=debate_desc4a,
        n_repeat=self.max_round
    )
    logs.append(log4a)

    debate_instruction4b = (
        "Sub-task 4b: Solve the mass-balance, charge-balance, and equilibrium equations simultaneously to determine the final [H+] concentration in solution, "
        "considering both leftover strong acid and H+ generated by Fe3+ hydrolysis. Carefully handle volume effects and avoid oversimplified pH assumptions."
    )
    final_decision_instruction4b = "Sub-task 4b: Provide the solved [H+] concentration and detailed reasoning."
    debate_desc4b = {
        'instruction': debate_instruction4b,
        'final_decision_instruction': final_decision_instruction4b,
        'input': [taskInfo, results3['answer'], results4a['answer']],
        'context_desc': ["user query", "answer of stage_2.subtask_3", "answer of stage_3.subtask_4a"],
        'temperature': 0.5
    }
    results4b, log4b = await self.debate(
        subtask_id="stage_3.subtask_4b",
        debate_desc=debate_desc4b,
        n_repeat=self.max_round
    )
    logs.append(log4b)

    cot_reflect_instruction4c = (
        "Sub-task 4c: Calculate the pH of the resulting solution from the [H+] concentration obtained in Subtask 4b. "
        "Ensure the pH reflects the combined effects of leftover acid and Fe3+ hydrolysis equilibria, correcting the previous assumption that pH depends solely on leftover strong acid."
    )
    critic_instruction4c = (
        "Please review and provide the limitations of provided pH calculation and ensure correctness considering all equilibria and volume effects."
    )
    cot_reflect_desc4c = {
        'instruction': cot_reflect_instruction4c,
        'critic_instruction': critic_instruction4c,
        'input': [taskInfo, results3['answer'], results4b['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "answer of stage_2.subtask_3", "answer of stage_3.subtask_4b"]
    }
    results4c, log4c = await self.reflexion(
        subtask_id="stage_3.subtask_4c",
        reflect_desc=cot_reflect_desc4c,
        n_repeat=self.max_round
    )
    logs.append(log4c)

    debate_instruction5 = (
        "Sub-task 5: Integrate results from previous subtasks to derive the final answers: the minimum volume of 0.1 M acid required to dissolve 0.1 g Fe(OH)3 in 100 cm³ total volume, "
        "and the pH of the resulting solution. Compare these results with the given multiple-choice options to select the correct answer. "
        "Ensure consistency and correctness by cross-checking all assumptions and calculations, avoiding premature or incorrect answer selection."
    )
    final_decision_instruction5 = "Sub-task 5: Provide final answer with explanation and choice selection."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results3['answer'], results4c['answer']],
        'context_desc': ["user query", "answer of stage_2.subtask_3", "answer of stage_3.subtask_4c"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="stage_4.subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
