async def forward_160(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Precisely define and distinguish the physical meanings of λ1 and λ2, "
        "explicitly identifying which species' mean free path each represents (gas molecules vs electrons) and the scattering processes involved, "
        "with context from the provided query about electron microscopy and vacuum conditions."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_s1, log_s1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log_s1)

    debate_instruction2 = (
        "Sub-task 2: Conduct a structured debate evaluating competing hypotheses about the impact of the electron beam on mean free path λ2 vs λ1. "
        "One argument posits the beam adds scatterers, reducing λ2 relative to λ1; the opposing argument emphasizes electrons have smaller scattering cross sections, implying λ2 >= λ1. "
        "Resolve the conceptual conflict with context from the query and output thinking and answer."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results_s1['thinking'], results_s1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_s2, log_s2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log_s2)

    cot_instruction3 = (
        "Sub-task 1 of Stage 2: Derive quantitatively the relationship between λ2 and λ1, incorporating the factor 1.22. "
        "Validate the physical and empirical origin of 1.22, clarify its directionality, and ensure consistency with electron-gas scattering cross sections, "
        "based on outputs from Stage 1 subtasks and the query context."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results_s1['thinking'], results_s1['answer'], results_s2['thinking'], results_s2['answer']],
        'temperature': 0.0,
        'context': [
            "user query",
            "thinking of stage_1.subtask_1",
            "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2",
            "answer of stage_1.subtask_2"
        ]
    }
    results_s3, log_s3 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log_s3)

    cot_reflect_instruction4 = (
        "Sub-task 1 of Stage 3: Integrate all prior definitions, debate outcomes, and quantitative derivations to determine the correct conclusion about λ2 relative to λ1. "
        "Select the best matching choice from the given options, ensuring alignment with clarified physical meanings and validated quantitative relationships, "
        "using Reflexion to critically evaluate and synthesize all inputs."
    )
    critic_instruction4 = (
        "Please review the integration and final conclusion for consistency and limitations."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [
            taskInfo,
            results_s1['thinking'], results_s1['answer'],
            results_s2['thinking'], results_s2['answer'],
            results_s3['thinking'], results_s3['answer']
        ],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': [
            "user query",
            "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2",
            "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"
        ]
    }
    results_s4, log_s4 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log_s4)

    final_answer = await self.make_final_answer(results_s4['thinking'], results_s4['answer'])
    return final_answer, logs
