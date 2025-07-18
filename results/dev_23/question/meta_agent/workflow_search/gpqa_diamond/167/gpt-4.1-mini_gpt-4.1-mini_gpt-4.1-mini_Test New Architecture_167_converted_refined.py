async def forward_167(self, taskInfo):
    logs = []

    debate_instruction_1 = (
        "Subtask 1: Analyze and classify each of the four issues (mutually incompatible data formats, 'chr' / 'no chr' confusion, "
        "reference assembly mismatch, incorrect ID conversion) in terms of their nature, causes, frequency, and how they can lead to difficult-to-spot erroneous results in genomics data analysis. "
        "Embed feedback to avoid downplaying the frequency and subtlety of issues 1 and 4, ensuring a balanced and comprehensive evaluation."
    )
    debate_desc_1 = {
        'instruction': debate_instruction_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ['user query'],
        'output': ['thinking', 'answer']
    }
    results1, log1 = await self.debate(
        subtask_id='subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    debate_instruction_2 = (
        "Subtask 2: Prioritize the four issues based on their frequency and subtlety as sources of difficult-to-spot errors, "
        "using the detailed analysis from Subtask 1. Explicitly address and document the relative importance of each issue, especially highlighting the 'chr' / 'no chr' confusion as a top source. "
        "Incorporate feedback to avoid misinterpretation or omission of key issues during prioritization."
    )
    debate_desc_2 = {
        'instruction': debate_instruction_2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'],
        'output': ['thinking', 'answer']
    }
    results2, log2 = await self.debate(
        subtask_id='subtask_2',
        debate_desc=debate_desc_2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction_3 = (
        "Subtask 3: Explicitly map each multiple-choice answer option to the exact set of numbered issues it includes, "
        "verifying that 'All of the above' unambiguously refers to issues 1 through 4. This step must prevent inconsistent assumptions about option contents and ensure clarity before answer selection. "
        "Embed feedback to enforce explicit verification of option semantics to avoid discarding correct answers due to misinterpretation."
    )
    cot_sc_desc_3 = {
        'instruction': cot_sc_instruction_3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2'],
        'output': ['thinking', 'answer']
    }
    results3, log3 = await self.sc_cot(
        subtask_id='subtask_3',
        cot_agent_desc=cot_sc_desc_3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction_4 = (
        "Subtask 4: Reconcile the prioritized issues with the mapped answer choices, explicitly identifying any discrepancies such as missing top-priority issues in the options. "
        "Reflect on the impact of these discrepancies on answer validity and discuss whether the provided options fully capture the correct answer. "
        "Incorporate a Reflexion step to self-check consistency between prioritization and answer selection, preventing acceptance of incomplete or inconsistent options as final answers."
    )
    cot_reflect_desc_4 = {
        'instruction': cot_reflect_instruction_4,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2', 'thinking of subtask 3', 'answer of subtask 3']
    }
    results4, log4 = await self.reflexion(
        subtask_id='subtask_4',
        reflect_desc=cot_reflect_desc_4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction_5 = (
        "Subtask 5: Select the best answer choice from the given options based on the reconciled evaluation and explicit verification steps. "
        "Provide a clear rationale that references the mapping and reconciliation subtasks, ensuring the final choice aligns with the prioritized issues and the verified option contents. "
        "If no option fully matches the prioritized issues, explicitly state this and justify the closest approximation chosen."
    )
    cot_sc_desc_5 = {
        'instruction': cot_sc_instruction_5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4'],
        'output': ['thinking', 'answer']
    }
    results5, log5 = await self.sc_cot(
        subtask_id='subtask_5',
        cot_agent_desc=cot_sc_desc_5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
