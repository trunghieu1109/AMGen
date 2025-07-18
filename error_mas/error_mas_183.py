async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze the role and effect of each reagent and reaction step on benzene and its substituted intermediates, "
        "focusing on how each step modifies the aromatic ring and influences subsequent substitutions, with context from the query: {question}. "
        "Reagents include tert-butyl chloride/AlCl3, HNO3/H2SO4, Fe/HCl, NaNO2/HCl, H3O+, NaOH/EtBr, SO3/H2SO4, and dilute H2SO4."
    ).format(question=taskInfo['question'])

    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }

    results1, log1 = await self.sc_cot(
        subtask_id='subtask_1',
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, determine the directing effects and regioselectivity of substituents introduced at each step, "
        "and predict the substitution pattern on the benzene ring after each reaction, with context from the query and previous analysis."
    )

    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for directing effects and regioselectivity analysis."
    )

    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }

    results2, log2 = await self.sc_cot(
        subtask_id='subtask_2',
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Based on outputs from Sub-tasks 1 and 2, identify the functional group transformations (e.g., nitro to amine, amine to diazonium, diazonium to phenol, phenol to ether) "
        "and their sequence dependencies to ensure compatibility and high yield, with context from the query and previous analyses."
    )

    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent answer for functional group transformations and sequence dependencies."
    )

    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2']
    }

    results3, log3 = await self.sc_cot(
        subtask_id='subtask_3',
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate each given reaction sequence option against the analyzed directing effects, substitution patterns, "
        "and functional group transformations from previous subtasks to assess feasibility and yield potential, with context from the query and prior analyses."
    )

    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent evaluation of the reaction sequences."
    )

    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'context_desc': ['user query', 'thinking of subtask 2', 'answer of subtask 2', 'thinking of subtask 3', 'answer of subtask 3'],
        'temperature': 0.5
    }

    results4, log4 = await self.debate(
        subtask_id='subtask_4',
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Prioritize and select the optimal reaction sequence that leads to the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene, "
        "providing a rationale based on mechanistic reasoning and synthetic strategy, with context from the query and previous evaluations."
    )

    final_decision_instruction5 = (
        "Sub-task 5: Provide the final selected optimal reaction sequence and rationale."
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
