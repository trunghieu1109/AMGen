async def forward_156(self, taskInfo):
    logs = []
    loop_results = {
        'stage_2.subtask_1': {'thinking': [], 'answer': []},
        'stage_3.subtask_1': {'thinking': [], 'answer': []},
        'stage_4.subtask_1': {'thinking': [], 'answer': []}
    }

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the outbreak context and identify key elements such as virus type (retrovirus), "
        "diagnostic goals, and candidate detection methods from the query, ensuring no assumptions about virus novelty are made. "
        "Input content includes the query question and all four choices provided."
    )
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id='stage_0.subtask_1',
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_1_1 = (
        "Sub-task 1: Determine whether the retrovirus genome is already characterized and reference sequences are available, "
        "explicitly addressing the previous failure of assuming de novo sequencing is always required. "
        "Input content includes results (thinking and answer) from stage_0.subtask_1."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    debate_instruction_1_2 = (
        "Sub-task 2: Based on the genome availability result from stage_1.subtask_1, plan the appropriate diagnostic assay design: "
        "if genome is known, proceed to design a one-step RT-qPCR assay; if unknown, plan for sequencing (e.g., metagenomic or targeted RNA sequencing) before assay design. "
        "This subtask must explicitly avoid the previous echo chamber by incorporating a reflexion or debate step to validate the chosen path. "
        "Input content includes results (thinking and answer) from stage_1.subtask_1."
    )
    final_decision_instruction_1_2 = (
        "Sub-task 2: Synthesize and choose the most consistent and justified assay design plan based on genome availability, "
        "avoiding assumptions and validating the approach."
    )
    debate_desc_1_2 = {
        'instruction': debate_instruction_1_2,
        'final_decision_instruction': final_decision_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
        'temperature': 0.5
    }
    results_1_2, log_1_2 = await self.debate(
        subtask_id='stage_1.subtask_2',
        debate_desc=debate_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    for iteration in range(2):
        cot_instruction_2_1 = (
            "Sub-task 1: Evaluate candidate diagnostic approaches (DNA sequencing + PCR, IgG ELISA, symptom-based nested PCR, cDNA sequencing + real-time PCR) "
            "against criteria of speed, accuracy, and retrovirus biology, incorporating the conditional path from stage_1.subtask_2. "
            "Explicitly incorporate feedback to avoid assuming sequencing necessity and to consider antibody detection limitations due to infection timing. "
            "Input content includes results (thinking and answer) from stage_1.subtask_2 and all previous iterations of stage_4.subtask_1."
        )
        cot_agent_desc_2_1 = {
            'instruction': cot_instruction_2_1,
            'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']] + loop_results['stage_4.subtask_1']['thinking'] + loop_results['stage_4.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2', 'thinking of previous stage_4.subtask_1 iterations', 'answer of previous stage_4.subtask_1 iterations']
        }
        results_2_1, log_2_1 = await self.cot(
            subtask_id='stage_2.subtask_1',
            cot_agent_desc=cot_agent_desc_2_1
        )
        logs.append(log_2_1)
        loop_results['stage_2.subtask_1']['thinking'].append(results_2_1['thinking'])
        loop_results['stage_2.subtask_1']['answer'].append(results_2_1['answer'])

        cot_reflect_instruction_3_1 = (
            "Sub-task 1: Refine the evaluation results from stage_2.subtask_1 to select the best candidate diagnostic method, "
            "ensuring the reasoning addresses previous logical breaks and context gaps by explicitly justifying the choice with respect to retrovirus biology and outbreak context. "
            "Input content includes results (thinking and answer) from stage_2.subtask_1."
        )
        critic_instruction_3_1 = (
            "Please review and provide the limitations of the selected diagnostic method and reasoning, "
            "ensuring no unfounded assumptions remain and the choice is well justified."
        )
        cot_reflect_desc_3_1 = {
            'instruction': cot_reflect_instruction_3_1,
            'critic_instruction': critic_instruction_3_1,
            'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
        }
        results_3_1, log_3_1 = await self.reflexion(
            subtask_id='stage_3.subtask_1',
            reflect_desc=cot_reflect_desc_3_1,
            n_repeat=self.max_round
        )
        logs.append(log_3_1)
        loop_results['stage_3.subtask_1']['thinking'].append(results_3_1['thinking'])
        loop_results['stage_3.subtask_1']['answer'].append(results_3_1['answer'])

        cot_reflect_instruction_4_1 = (
            "Sub-task 1: Assess the validity and suitability of the selected diagnostic method for retrovirus detection, "
            "providing feedback for refinement and ensuring no unfounded assumptions remain. "
            "This subtask should close the loop by feeding back into stage_2.subtask_1 for iterative improvement if needed. "
            "Input content includes results (thinking and answer) from stage_3.subtask_1."
        )
        critic_instruction_4_1 = (
            "Please review the selected diagnostic method and provide any necessary refinements or confirm its suitability, "
            "ensuring the solution is robust and well-grounded."
        )
        cot_reflect_desc_4_1 = {
            'instruction': cot_reflect_instruction_4_1,
            'critic_instruction': critic_instruction_4_1,
            'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
        }
        results_4_1, log_4_1 = await self.reflexion(
            subtask_id='stage_4.subtask_1',
            reflect_desc=cot_reflect_desc_4_1,
            n_repeat=self.max_round
        )
        logs.append(log_4_1)
        loop_results['stage_4.subtask_1']['thinking'].append(results_4_1['thinking'])
        loop_results['stage_4.subtask_1']['answer'].append(results_4_1['answer'])

    final_answer = await self.make_final_answer(loop_results['stage_4.subtask_1']['thinking'][-1], loop_results['stage_4.subtask_1']['answer'][-1])
    return final_answer, logs
