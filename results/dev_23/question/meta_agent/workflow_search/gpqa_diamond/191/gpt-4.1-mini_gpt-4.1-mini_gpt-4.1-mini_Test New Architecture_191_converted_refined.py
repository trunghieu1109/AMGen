async def forward_191(self, taskInfo):
    logs = []

    cot_instruction_s1_st1 = (
        "Sub-task 1: Analyze and classify the physical setup and given parameters: the spherical conductor, "
        "the off-center cavity, the charge +q inside the cavity, and the position vectors and angles (s, l, L, θ). "
        "Explicitly clarify the directions and definitions of vectors s and l, and the geometric relationships among R, r, s, l, L, and θ. "
        "This subtask addresses the previous confusion about vector definitions and ensures a clear coordinate framework for subsequent analysis."
    )
    cot_agent_desc_s1_st1 = {
        'instruction': cot_instruction_s1_st1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_s1_st1, log_s1_st1 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_s1_st1
    )
    logs.append(log_s1_st1)

    debate_instruction_s1_st2 = (
        "Sub-task 2: Assess the electrostatic consequences of placing a charge +q inside the off-center cavity within the uncharged spherical conductor. "
        "Specifically, analyze the induced charge distributions on the cavity surface and the conductor's outer surface, emphasizing the breaking of spherical symmetry due to the off-center displacement s. "
        "This subtask must explicitly address the failure in previous attempts that ignored multipole moments and assumed spherical symmetry."
    )
    debate_desc_s1_st2 = {
        'instruction': debate_instruction_s1_st2,
        'context': ['user query', results_s1_st1['thinking'], results_s1_st1['answer']],
        'input': [taskInfo, results_s1_st1['thinking'], results_s1_st1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results_s1_st2, log_s1_st2 = await self.debate(
        subtask_id='stage_1.subtask_2',
        debate_desc=debate_desc_s1_st2,
        n_repeat=self.max_round
    )
    logs.append(log_s1_st2)

    debate_instruction_s2_st1 = (
        "Sub-task 1: Derive the expression for the external electric field at point P (distance L from the conductor center) considering the induced charges and the off-center cavity charge. "
        "This includes performing a multipole expansion of the induced charge distribution on the conductor's outer surface, capturing the angular dependence on θ and the displacement s. "
        "The derivation must avoid oversimplification to a monopole field and explicitly incorporate the geometry parameters (s, θ, l, L)."
    )
    debate_desc_s2_st1 = {
        'instruction': debate_instruction_s2_st1,
        'context': ['user query', results_s1_st1['thinking'], results_s1_st1['answer'], results_s1_st2['thinking'], results_s1_st2['answer']],
        'input': [taskInfo, results_s1_st2['thinking'], results_s1_st2['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results_s2_st1, log_s2_st1 = await self.debate(
        subtask_id='stage_2.subtask_1',
        debate_desc=debate_desc_s2_st1,
        n_repeat=self.max_round
    )
    logs.append(log_s2_st1)

    cot_sc_instruction_s2_st2 = (
        "Sub-task 2: Express the derived electric field magnitude entirely in terms of the given problem variables L, l, s, and θ, "
        "explicitly substituting any generic or undefined distances (e.g., r) with these variables. "
        "This step addresses the critical failure in previous attempts where the final expression was left in terms of undefined or confused variables, leading to incorrect answer selection."
    )
    cot_sc_desc_s2_st2 = {
        'instruction': cot_sc_instruction_s2_st2,
        'input': [taskInfo, results_s2_st1['thinking'], results_s2_st1['answer']],
        'temperature': 0.5,
        'context': ['user query', results_s2_st1['thinking'], results_s2_st1['answer']]
    }
    results_s2_st2, log_s2_st2 = await self.sc_cot(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_sc_desc_s2_st2,
        n_repeat=self.max_sc
    )
    logs.append(log_s2_st2)

    cot_reflect_instruction_s3_st1 = (
        "Sub-task 1: Critically verify and match the final derived expression for the electric field magnitude at point P against the provided multiple-choice options. "
        "This subtask must ensure that the expression uses the correct variables (especially L) and correctly reflects the physical scenario including the off-center cavity effects. "
        "It should explicitly rule out incorrect options that ignore the geometry or use wrong variables, preventing the previous mistake of selecting option A instead of the correct option D."
    )
    cot_reflect_desc_s3_st1 = {
        'instruction': cot_reflect_instruction_s3_st1,
        'input': [taskInfo, results_s2_st2['thinking'], results_s2_st2['answer'], results_s1_st1['thinking']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', results_s2_st2['thinking'], results_s2_st2['answer'], results_s1_st1['thinking']]
    }
    results_s3_st1, log_s3_st1 = await self.reflexion(
        subtask_id='stage_3.subtask_1',
        reflect_desc=cot_reflect_desc_s3_st1,
        n_repeat=self.max_round
    )
    logs.append(log_s3_st1)

    final_answer = await self.make_final_answer(results_s3_st1['thinking'], results_s3_st1['answer'])
    return final_answer, logs
