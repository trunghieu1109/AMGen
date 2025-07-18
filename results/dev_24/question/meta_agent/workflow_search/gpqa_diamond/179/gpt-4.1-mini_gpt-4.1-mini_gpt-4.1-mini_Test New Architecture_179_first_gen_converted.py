async def forward_179(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and summarize all given information, constants, and constraints relevant to the problem, "
        "including charge values, geometry, and what is being asked."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent and correct summary of the problem information."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze the physical and geometric relationships between the charges, including the nature of interactions "
        "and constraints on their positions, to frame the problem mathematically."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct mathematical framing of the problem."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Compute the electrostatic potential energy contributions: (a) between the central charge and each of the 12 charges on the sphere, "
        "and (b) among the 12 charges themselves, considering the minimum energy configuration on the sphere."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Decide on the most accurate and justified calculation of the electrostatic potential energy contributions."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Derive the total minimum electrostatic potential energy of the system by summing all pairwise interactions "
        "and applying known results or approximations for the minimum energy configuration of 12 identical charges on a sphere."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of provided solutions for this energy derivation problem."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'critic_instruction': critic_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context_desc': [
            "user query", "thinking of subtask 1", "answer of subtask 1",
            "thinking of subtask 2", "answer of subtask 2",
            "thinking of subtask 3", "answer of subtask 3"
        ]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Compare the computed minimum energy value with the provided answer choices and select the correct one, "
        "ensuring the result is expressed in Joules and rounded to three decimals as requested."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Synthesize and select the final answer choice that best matches the computed minimum energy."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
