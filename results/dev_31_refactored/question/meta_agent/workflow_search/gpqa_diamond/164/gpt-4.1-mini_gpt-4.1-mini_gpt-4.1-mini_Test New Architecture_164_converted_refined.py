async def forward_164(self, taskInfo):
    logs = []

    results = {}

    # Stage 0: Extract and categorize relevant information from the query

    # stage_0.subtask_1: Identify and extract key elements from the query, including catalyst types, activators, polymerization conditions, and the four provided statements, ensuring clarity on ambiguous terms such as 'group VIa transition metal' and 'aluminum-based activators'.
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and extract key elements from the query, including catalyst types, activators, polymerization conditions, "
        "and the four provided statements, ensuring clarity on ambiguous terms such as 'group VIa transition metal' and 'aluminum-based activators'. "
        "Input content: taskInfo"
    )
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)
    results['stage_0.subtask_1'] = results_0_1

    # stage_0.subtask_2: Analyze the extracted statements to explicitly enforce the multiple-choice constraint that exactly one statement is correct; identify any direct contradictions or mutual exclusivity among the four statements to guide subsequent evaluation and avoid assuming multiple truths.
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, analyze the extracted statements to explicitly enforce the multiple-choice constraint "
        "that exactly one statement is correct; identify any direct contradictions or mutual exclusivity among the four statements to guide subsequent evaluation and avoid assuming multiple truths. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1"
    )
    final_decision_instruction_0_2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer enforcing the single-correct-answer constraint for the multiple-choice statements."
    )
    cot_sc_desc_0_2 = {
        'instruction': cot_sc_instruction_0_2,
        'final_decision_instruction': final_decision_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.sc_cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_sc_desc_0_2, n_repeat=self.max_sc)
    logs.append(log_0_2)
    results['stage_0.subtask_2'] = results_0_2

    # Stage 1: Mechanistic and industrial feasibility evaluation of each statement

    # stage_1.subtask_1: For each of the four statements, perform a detailed mechanistic verification based on literature and known polymerization chemistry to determine if the catalyst-activator combinations can produce regular branches from ethylene alone, avoiding unchecked assumptions and ensuring alignment with the question's precise requirements.
    cot_sc_instruction_1_1 = (
        "Sub-task 1: For each of the four statements, perform a detailed mechanistic verification based on literature and known polymerization chemistry "
        "to determine if the catalyst-activator combinations can produce regular branches from ethylene alone, avoiding unchecked assumptions and ensuring alignment with the question's precise requirements. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1 and stage_0.subtask_2"
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize mechanistic verification results for all four statements with self-consistency."
    )
    cot_sc_desc_1_1 = {
        'instruction': cot_sc_instruction_1_1,
        'final_decision_instruction': final_decision_instruction_1_1,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_1_1, log_1_1 = await self.sc_cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_sc_desc_1_1, n_repeat=self.max_sc)
    logs.append(log_1_1)
    results['stage_1.subtask_1'] = results_1_1

    # stage_1.subtask_2: Assess the industrial implementation status and economic feasibility of the catalyst systems mentioned in the statements, including cost considerations of noble metal catalysts and the practical use of aluminum-based activators, to validate or refute claims about industrial scale and activator effectiveness.
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Assess the industrial implementation status and economic feasibility of the catalyst systems mentioned in the statements, "
        "including cost considerations of noble metal catalysts and the practical use of aluminum-based activators, to validate or refute claims about industrial scale and activator effectiveness. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1 and stage_0.subtask_2"
    )
    final_decision_instruction_1_2 = (
        "Sub-task 2: Synthesize industrial feasibility and economic assessment results with self-consistency."
    )
    cot_sc_desc_1_2 = {
        'instruction': cot_sc_instruction_1_2,
        'final_decision_instruction': final_decision_instruction_1_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_1_2, log_1_2 = await self.sc_cot(subtask_id='stage_1.subtask_2', cot_agent_desc=cot_sc_desc_1_2, n_repeat=self.max_sc)
    logs.append(log_1_2)
    results['stage_1.subtask_2'] = results_1_2

    # Stage 2: Analyze relationships and dependencies among catalysts, activators, and polymerization steps

    # stage_2.subtask_1: Integrate mechanistic and industrial feasibility results to determine functional associations and constraints between catalyst systems and activators for producing regularly branched polymers from ethylene, ensuring that the analysis explicitly addresses the essential additional reaction step and catalyst-activator compatibility.
    cot_instruction_2_1 = (
        "Sub-task 1: Integrate mechanistic and industrial feasibility results to determine functional associations and constraints between catalyst systems and activators "
        "for producing regularly branched polymers from ethylene, ensuring that the analysis explicitly addresses the essential additional reaction step and catalyst-activator compatibility. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1, stage_0.subtask_2, stage_1.subtask_1, and stage_1.subtask_2"
    )
    cot_agent_desc_2_1 = {
        'instruction': cot_instruction_2_1,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer'], results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)
    results['stage_2.subtask_1'] = results_2_1

    # Loop control flow: 2 iterations over stage 3 subtasks
    loop_results = {
        'stage_3.subtask_1': {'thinking': [], 'answer': []},
        'stage_3.subtask_2': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        # stage_3.subtask_1: Produce structured intermediate reasoning and provisional conclusions about which single statement is correct, synthesizing all prior analyses while explicitly considering the mutual exclusivity constraint and avoiding the error of accepting multiple statements as true.
        cot_instruction_3_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Produce structured intermediate reasoning and provisional conclusions about which single statement is correct, "
            "synthesizing all prior analyses while explicitly considering the mutual exclusivity constraint and avoiding the error of accepting multiple statements as true. "
            "Input content: taskInfo, thinking and answer from stage_2.subtask_1"
        )
        cot_agent_desc_3_1 = {
            'instruction': cot_instruction_3_1,
            'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']] + loop_results['stage_3.subtask_1']['thinking'] + loop_results['stage_3.subtask_1']['answer'] + loop_results['stage_3.subtask_2']['thinking'] + loop_results['stage_3.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1'] + ['thinking of previous iterations stage_3.subtask_1']*len(loop_results['stage_3.subtask_1']['thinking']) + ['answer of previous iterations stage_3.subtask_1']*len(loop_results['stage_3.subtask_1']['answer']) + ['thinking of previous iterations stage_3.subtask_2']*len(loop_results['stage_3.subtask_2']['thinking']) + ['answer of previous iterations stage_3.subtask_2']*len(loop_results['stage_3.subtask_2']['answer'])
        }
        results_3_1, log_3_1 = await self.cot(subtask_id='stage_3.subtask_1', cot_agent_desc=cot_agent_desc_3_1)
        logs.append(log_3_1)
        loop_results['stage_3.subtask_1']['thinking'].append(results_3_1['thinking'])
        loop_results['stage_3.subtask_1']['answer'].append(results_3_1['answer'])

        # stage_3.subtask_2: Critically review the provisional conclusions to challenge any multi-valid-answer outcomes, enforce the single-correct-answer rule, and identify contradictions or inconsistencies, thereby refining the reasoning towards a unique best answer.
        cot_reflect_instruction_3_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Critically review the provisional conclusions to challenge any multi-valid-answer outcomes, "
            "enforce the single-correct-answer rule, and identify contradictions or inconsistencies, thereby refining the reasoning towards a unique best answer. "
            "Input content: taskInfo, thinking and answer from stage_3.subtask_1"
        )
        critic_instruction_3_2 = (
            "Please review and provide the limitations of provided solutions of stage_3.subtask_1, focusing on enforcing the single-correct-answer rule and identifying contradictions or inconsistencies."
        )
        cot_reflect_desc_3_2 = {
            'instruction': cot_reflect_instruction_3_2,
            'critic_instruction': critic_instruction_3_2,
            'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']] + loop_results['stage_3.subtask_2']['thinking'] + loop_results['stage_3.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1'] + ['thinking of previous iterations stage_3.subtask_2']*len(loop_results['stage_3.subtask_2']['thinking']) + ['answer of previous iterations stage_3.subtask_2']*len(loop_results['stage_3.subtask_2']['answer'])
        }
        results_3_2, log_3_2 = await self.reflexion(subtask_id='stage_3.subtask_2', reflect_desc=cot_reflect_desc_3_2, n_repeat=self.max_round)
        logs.append(log_3_2)
        loop_results['stage_3.subtask_2']['thinking'].append(results_3_2['thinking'])
        loop_results['stage_3.subtask_2']['answer'].append(results_3_2['answer'])

    # Stage 4: Select the best candidate statement based on evaluation and reasoning

    # stage_4.subtask_1: Evaluate candidate statements against all mechanistic, industrial, and critical review criteria to select the most accurate and consistent single statement as the final answer, ensuring no ambiguity or multiple selections.
    cot_instruction_4_1 = (
        "Sub-task 1: Evaluate candidate statements against all mechanistic, industrial, and critical review criteria to select the most accurate and consistent single statement as the final answer, "
        "ensuring no ambiguity or multiple selections. "
        "Input content: taskInfo, thinking and answer from last iteration of stage_3.subtask_2"
    )
    cot_agent_desc_4_1 = {
        'instruction': cot_instruction_4_1,
        'input': [taskInfo, loop_results['stage_3.subtask_2']['thinking'][-1], loop_results['stage_3.subtask_2']['answer'][-1]],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of last iteration stage_3.subtask_2', 'answer of last iteration stage_3.subtask_2']
    }
    results_4_1, log_4_1 = await self.cot(subtask_id='stage_4.subtask_1', cot_agent_desc=cot_agent_desc_4_1)
    logs.append(log_4_1)
    results['stage_4.subtask_1'] = results_4_1

    # Stage 5: Assess the validity of the selected statement

    # stage_5.subtask_1: Apply a systematic debate-style evaluation to confirm the correctness and exclusivity of the selected statement, considering alternative viewpoints and potential counterarguments to prevent premature or incorrect conclusions.
    debate_instruction_5_1 = (
        "Sub-task 1: Apply a systematic debate-style evaluation to confirm the correctness and exclusivity of the selected statement, "
        "considering alternative viewpoints and potential counterarguments to prevent premature or incorrect conclusions. "
        "Input content: taskInfo, thinking and answer from stage_4.subtask_1"
    )
    final_decision_instruction_5_1 = (
        "Sub-task 1: Confirm the correctness and exclusivity of the selected statement through debate."
    )
    debate_desc_5_1 = {
        'instruction': debate_instruction_5_1,
        'final_decision_instruction': final_decision_instruction_5_1,
        'input': [taskInfo, results_4_1['thinking'], results_4_1['answer']],
        'context_desc': ['user query', 'thinking of stage_4.subtask_1', 'answer of stage_4.subtask_1'],
        'temperature': 0.5
    }
    results_5_1, log_5_1 = await self.debate(subtask_id='stage_5.subtask_1', debate_desc=debate_desc_5_1, n_repeat=self.max_round)
    logs.append(log_5_1)
    results['stage_5.subtask_1'] = results_5_1

    # Stage 6: Consolidate and refine the final answer

    # stage_6.subtask_1: Simplify and consolidate all evaluation results to produce a clear, concise final answer that explicitly states which single statement is correct, incorporating insights from the debate and ensuring compliance with the question's requirement for a unique correct choice.
    cot_instruction_6_1 = (
        "Sub-task 1: Simplify and consolidate all evaluation results to produce a clear, concise final answer that explicitly states which single statement is correct, "
        "incorporating insights from the debate and ensuring compliance with the question's requirement for a unique correct choice. "
        "Input content: taskInfo, thinking and answer from stage_4.subtask_1 and stage_5.subtask_1"
    )
    cot_agent_desc_6_1 = {
        'instruction': cot_instruction_6_1,
        'input': [taskInfo, results_4_1['thinking'], results_4_1['answer'], results_5_1['thinking'], results_5_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_4.subtask_1', 'answer of stage_4.subtask_1', 'thinking of stage_5.subtask_1', 'answer of stage_5.subtask_1']
    }
    results_6_1, log_6_1 = await self.review(subtask_id='stage_6.subtask_1', review_desc=cot_agent_desc_6_1)
    logs.append(log_6_1)
    results['stage_6.subtask_1'] = results_6_1

    final_answer = await self.make_final_answer(results_6_1['thinking'], results_6_1['answer'])
    return final_answer, logs
