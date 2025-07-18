async def forward_156(self, taskInfo):
    logs = []

    cot_sc_instruction0 = (
        "Sub-task 1: Analyze and classify the key elements involved in the diagnostic design problem, "
        "including the nature of the retrovirus, possible molecular targets (DNA, RNA, antibodies), "
        "and diagnostic methods (PCR variants, ELISA), based on the given choices and biological context."
    )
    final_decision_instruction0 = (
        "Sub-task 1: Synthesize and choose the most consistent and correct classification of key elements "
        "for the diagnostic design problem given all the above thinking and answers."
    )
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'final_decision_instruction': final_decision_instruction0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results0, log0 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    logs.append(log0)

    debate_instruction1 = (
        "Sub-task 1: Assess the impact of different virus identification approaches (DNA sequencing, cDNA sequencing, "
        "antibody detection, symptom-based identification) on the choice of molecular target and diagnostic assay type, "
        "considering the retrovirus biology and diagnostic requirements."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Debate and decide the best virus identification approach and its impact on diagnostic assay choice "
        "based on the analysis from stage_0.subtask_1."
    )
    debate_desc1 = {
        'instruction': debate_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Derive the optimal diagnostic strategy by selecting the most appropriate virus identification method "
        "and corresponding molecular diagnostic assay (e.g., real-time PCR targeting cDNA) that ensures quick and accurate detection of the retrovirus, "
        "based on the debate outcomes and previous analysis."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct optimal diagnostic strategy "
        "given all previous thinking and answers."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Combine and transform the insights from virus biology, molecular targets, and assay types "
        "to design a detailed workflow for the molecular diagnostic kit development, including sample type, assay protocol, "
        "and validation steps to ensure sensitivity and specificity."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of the proposed molecular diagnostic kit design workflow, "
        "and suggest improvements to ensure quick and accurate detection of the retrovirus."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'critic_instruction': critic_instruction3,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context_desc': [
            "user query",
            "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1",
            "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1",
            "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"
        ]
    }
    results3, log3 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
