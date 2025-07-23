async def forward_156(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_1 = (
            "Sub-task 1: Confirm and explicitly document that the retrovirus has an RNA genome and that reverse transcription to cDNA is mandatory before any sequencing or PCR steps. "
            "This gating step prevents ambiguity between DNA and RNA sequencing approaches and addresses previous logical flaws. "
            "Input content are results (both thinking and answer) from: none."
        )
        cot_agent_desc_1 = {
            'instruction': cot_instruction_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_1, log_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_1['answer'])
        logs.append(log_1)

        cot_instruction_2 = (
            "Sub-task 2: Retrieve consensus retroviral genome sequences from public databases to serve as the basis for primer and probe design, "
            "eliminating the impractical step of de novo sequencing of outbreak samples. This step addresses the failure of over-emphasizing cDNA sequencing of patient samples. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1."
        )
        cot_agent_desc_2 = {
            'instruction': cot_instruction_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_2, log_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_2['answer'])
        logs.append(log_2)

        cot_sc_instruction_3 = (
            "Sub-task 3: Design primers and probes specific to the retrovirus genome using the consensus sequences, ensuring compatibility with one-step real-time RT-PCR. "
            "This subtask depends on confirmed genome type and retrieved sequences to maintain logical flow and avoid unused outputs. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_0.subtask_2 & former iterations of stage_0.subtask_3."
        )
        final_decision_instruction_3 = (
            "Sub-task 3: Synthesize and choose the most consistent primer and probe design for the retrovirus molecular diagnostic kit."
        )
        cot_sc_desc_3 = {
            'instruction': cot_sc_instruction_3,
            'final_decision_instruction': final_decision_instruction_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.5,
            'context': [
                'user query',
                'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1',
                'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2',
                'thinking of previous iterations of stage_0.subtask_3', 'answer of previous iterations of stage_0.subtask_3'
            ]
        }
        results_3, log_3 = await self.sc_cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_sc_desc_3,
            n_repeat=self.max_sc
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_3['answer'])
        logs.append(log_3)

        cot_instruction_4 = (
            "Sub-task 4: Develop a one-step real-time RT-PCR assay protocol incorporating the designed primers and probes, explicitly including internal extraction and amplification controls, "
            "and plan a limit-of-detection (LOD) study using synthetic RNA standards. This addresses previous omissions of critical practical details for clinical deployment. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_3 & former iterations of stage_0.subtask_4."
        )
        cot_agent_desc_4 = {
            'instruction': cot_instruction_4,
            'input': loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context': [
                'user query',
                'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3',
                'thinking of previous iterations of stage_0.subtask_4', 'answer of previous iterations of stage_0.subtask_4'
            ]
        }
        results_4, log_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_4['answer'])
        logs.append(log_4)

        cot_reflect_instruction_5 = (
            "Sub-task 5: Perform a Reflexion and Quality Assurance review of the assay design to verify that all protocol steps (RNA extraction, reverse transcription, amplification, and analysis) include appropriate internal controls, "
            "defined acceptance criteria, and documentation for practical deployment. This subtask explicitly addresses previous feedback about missing controls and validation details. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_4."
        )
        critic_instruction_5 = (
            "Please review and provide the limitations of provided solutions of the one-step real-time RT-PCR assay design, focusing on internal controls, LOD, and clinical deployment readiness."
        )
        cot_reflect_desc_5 = {
            'instruction': cot_reflect_instruction_5,
            'critic_instruction': critic_instruction_5,
            'input': loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context': [
                'user query',
                'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4'
            ]
        }
        results_5, log_5 = await self.reflexion(
            subtask_id='stage_0.subtask_5',
            reflect_desc=cot_reflect_desc_5,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_5['answer'])
        logs.append(log_5)

    cot_reflect_instruction_1_1 = (
        "Sub-task 1: Review and simplify the intermediate assay design documentation to enhance clarity, usability, and ensure all critical elements (controls, LOD, one-step RT-PCR) are clearly described. "
        "This step uses outputs from the Reflexion QA subtask to ensure alignment with clinical standards. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_5."
    )
    cot_reflect_desc_1_1 = {
        'instruction': cot_reflect_instruction_1_1,
        'input': loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
    }
    results_1_1, log_1_1 = await self.reflexion(
        subtask_id='stage_1.subtask_1',
        reflect_desc=cot_reflect_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    debate_instruction_1_2 = (
        "Sub-task 2: Evaluate the developed one-step real-time RT-PCR assay against criteria of speed, accuracy, feasibility, and compliance with clinical diagnostic standards, "
        "explicitly considering internal controls and LOD results. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_5."
    )
    final_decision_instruction_1_2 = (
        "Sub-task 2: Select the best diagnostic approach based on evaluation criteria, prioritizing scientific soundness and clinical deployability."
    )
    debate_desc_1_2 = {
        'instruction': debate_instruction_1_2,
        'final_decision_instruction': final_decision_instruction_1_2,
        'input': loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'],
        'temperature': 0.5
    }
    results_1_2, log_1_2 = await self.debate(
        subtask_id='stage_1.subtask_2',
        debate_desc=debate_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    aggregate_instruction_1_3 = (
        "Sub-task 3: Select the best diagnostic approach based on the evaluation, prioritizing the one-step real-time RT-PCR assay with internal controls and validated LOD, "
        "ensuring the approach is scientifically sound and clinically deployable. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_1.subtask_2."
    )
    aggregate_desc_1_3 = {
        'instruction': aggregate_instruction_1_3,
        'input': [taskInfo] + [results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_1.subtask_1', 'solutions generated from stage_1.subtask_2']
    }
    results_1_3, log_1_3 = await self.aggregate(
        subtask_id='stage_1.subtask_3',
        aggregate_desc=aggregate_desc_1_3
    )
    logs.append(log_1_3)

    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Translate the selected diagnostic design into a detailed molecular kit development protocol, including reagent preparation, assay workflow, control implementation, and documentation for manufacturing and clinical use. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_3."
    )
    cot_reflect_desc_2_1 = {
        'instruction': cot_reflect_instruction_2_1,
        'input': [results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_1, log_2_1 = await self.reflexion(
        subtask_id='stage_2.subtask_1',
        reflect_desc=cot_reflect_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Simulate or model the one-step real-time RT-PCR assay performance to predict diagnostic accuracy, speed, and robustness, "
        "using data from LOD studies and control assays to validate assay reliability. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_3."
    )
    final_decision_instruction_2_2 = (
        "Sub-task 2: Synthesize simulation results to confirm assay reliability and performance metrics."
    )
    cot_sc_desc_2_2 = {
        'instruction': cot_sc_instruction_2_2,
        'final_decision_instruction': final_decision_instruction_2_2,
        'input': [results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_2, log_2_2 = await self.sc_cot(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_sc_desc_2_2,
        n_repeat=self.max_sc
    )
    logs.append(log_2_2)

    review_instruction_3_1 = (
        "Sub-task 1: Assess the developed molecular diagnostic kit protocol for compliance with clinical diagnostic standards, including verification of internal controls, assay sensitivity, specificity, and documentation completeness. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1 & stage_2.subtask_2."
    )
    review_desc_3_1 = {
        'instruction': review_instruction_3_1,
        'input': [results_2_1['thinking'], results_2_1['answer'], results_2_2['thinking'], results_2_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2']
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    cot_instruction_3_2 = (
        "Sub-task 2: Provide feedback on potential improvements or confirm readiness for practical deployment, ensuring the diagnostic kit meets all necessary criteria for outbreak response and clinical use. "
        "Input content are results (both thinking and answer) from: stage_3.subtask_1."
    )
    cot_agent_desc_3_2 = {
        'instruction': cot_instruction_3_2,
        'input': [results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    results_3_2, log_3_2 = await self.cot(
        subtask_id='stage_3.subtask_2',
        cot_agent_desc=cot_agent_desc_3_2
    )
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2['thinking'], results_3_2['answer'])
    return final_answer, logs
