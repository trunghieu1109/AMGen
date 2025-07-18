async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze the target molecule's substitution pattern and explicitly identify all required functional groups and their precise positions on the benzene ring. Emphasize the necessity to consider the synthetic feasibility of each substituent's introduction from benzene, avoiding assumptions of direct substitution for complex groups like ethoxy."
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

    debate_instruction2 = "Sub-task 2: Construct a detailed reagent-to-transformation mapping table that assigns each reagent or reagent pair in the given options to its specific chemical transformation and intermediate structure formation. This includes mapping benzene → nitrobenzene (HNO3/H2SO4), nitrobenzene → aniline (Fe/HCl), aniline → diazonium salt (NaNO2/HCl), diazonium salt → phenol (H3O+, heat), phenol → anisole (NaOH/EtBr), and other relevant steps such as Friedel-Crafts alkylation and sulfonation. This subtask must explicitly identify all intermediates and ensure no reagent is left unassigned or misused, addressing the previous failure of collapsing multi-step transformations into single steps."
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: Evaluate the chemical compatibility, order, and regioselectivity of the reaction steps in each provided sequence option, using the detailed reagent-to-transformation mappings and intermediate structures from previous subtasks. Critically assess whether the sequence logically and feasibly leads to the target molecule with correct substitution patterns, avoiding oversimplified or chemically impossible assumptions."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = "Sub-task 4: Assess the potential yield, practicality, and synthetic efficiency of each reaction sequence option based on known chemical principles, possible side reactions, isomer formation, and stability of intermediates. Incorporate considerations of reaction conditions and reagent compatibility to identify sequences likely to afford high yield and selectivity."
    critic_instruction4 = "Please review the assessment of yield, practicality, and synthetic efficiency and provide its limitations."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Select the optimal sequence of reactions from the given options that leads to the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene. Justify the choice based on the detailed synthetic feasibility, regioselectivity, and yield assessments from previous subtasks, ensuring all reagents and steps are accounted for and correctly assigned."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
