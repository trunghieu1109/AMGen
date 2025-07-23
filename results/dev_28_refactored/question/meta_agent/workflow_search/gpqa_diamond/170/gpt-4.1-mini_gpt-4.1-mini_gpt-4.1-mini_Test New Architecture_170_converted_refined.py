async def forward_170(self, taskInfo):
    logs = []

    cot_instruction1 = (
        'Sub-task 1: Interpret and clarify the meaning of the phrase "only one monobromo derivative is formed" in the context of electrophilic substitution on benzene derivatives. Explicitly determine how this constraint affects the possible isomer distributions (ortho, meta, para) for different substituent types (ortho/para directors vs meta directors). This subtask addresses previous errors caused by misinterpretation of this phrase, ensuring subsequent analysis correctly models isomer yields.'
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query'],
    }
    results1, log1 = await self.debate(
        subtask_id='subtask_1',
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        'Sub-task 2: Analyze the electronic and steric effects of each substituent on the benzene ring to determine their directing effects (ortho/para or meta) and activation/deactivation strength influencing electrophilic substitution regioselectivity. This includes classifying substituents as activating or deactivating and their typical directing behavior. Avoid oversimplifications and explicitly note any assumptions. This subtask builds foundational knowledge for para-isomer yield estimation.'
    )
    final_decision_instruction2 = (
        'Sub-task 2: Synthesize and choose the most consistent answer for the analysis of substituent effects and directing behavior.'
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1'],
    }
    results2, log2 = await self.sc_cot(
        subtask_id='subtask_2',
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        'Sub-task 3: Perform a detailed comparative analysis of closely related substituents, especially the meta directors benzoic acid (–COOH) and ethyl benzoate (–COOC2H5). Incorporate authoritative chemical principles, experimental data, or literature references to resolve subtle differences in their para-isomer yields and deactivation strengths. This subtask addresses the critical failure of misordering these substituents in previous attempts and prevents reliance on unverified assumptions.'
    )
    final_decision_instruction3 = (
        'Sub-task 3: Synthesize and choose the most consistent answer for the comparative analysis of benzoic acid and ethyl benzoate substituents.'
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of subtask 2', 'answer of subtask 2'],
    }
    results3, log3 = await self.sc_cot(
        subtask_id='subtask_3',
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        'Sub-task 4: Integrate the insights from previous subtasks to derive the expected relative weight fractions of the para-isomer for each substance. Consider directing effects, activation/deactivation strength, steric hindrance, and clarified isomer distribution constraints. Explicitly document the rationale for the ordering, ensuring internal consistency and alignment with chemical principles. This subtask avoids confirmation bias by requiring justification rather than fitting to given choices.'
    )
    critic_instruction4 = (
        'Please review and provide the limitations of provided solutions of para-isomer yield ordering and rationale.'
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'critic_instruction': critic_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context_desc': [
            'user query',
            'thinking of subtask 1', 'answer of subtask 1',
            'thinking of subtask 2', 'answer of subtask 2',
            'thinking of subtask 3', 'answer of subtask 3'
        ],
    }
    results4, log4 = await self.reflexion(
        subtask_id='subtask_4',
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        'Sub-task 5: Critically evaluate the derived para-isomer yield order against the multiple-choice options provided. Enforce exact matching of the established order with one of the given choices; if no exact match is found, trigger a re-evaluation of the analysis rather than selecting the closest fit. This prevents errors from forcing a choice and ensures the final answer is robust and justified.'
    )
    final_decision_instruction5 = (
        'Sub-task 5: Select the exact matching multiple-choice option for the para-isomer yield order or trigger re-evaluation.'
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'context_desc': ['user query', 'thinking of subtask 4', 'answer of subtask 4'],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id='subtask_5',
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
