async def forward_160(self, taskInfo):
    logs = []

    cot_sc_instruction_0_1 = (
        "Sub-task 0_1: Extract and summarize all given information from the query, including system parameters, vacuum conditions, initial mean free path λ1, and observed change to λ2 upon electron beam initiation."
    )
    cot_sc_desc_0_1 = {
        'instruction': cot_sc_instruction_0_1,
        'final_decision_instruction': "Sub-task 0_1: Synthesize and choose the most consistent summary of given information.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results_0_1, log_0_1 = await self.sc_cot(
        subtask_id="subtask_0_1",
        cot_agent_desc=cot_sc_desc_0_1,
        n_repeat=self.max_sc
    )
    logs.append(log_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 0_2: Clarify definitions and assumptions related to mean free path λ1 and λ2, including the physical meaning of λ2 in the context of electron scattering and the significance of the factor 1.22."
    )
    cot_sc_desc_0_2 = {
        'instruction': cot_sc_instruction_0_2,
        'final_decision_instruction': "Sub-task 0_2: Synthesize and choose the most consistent clarification of definitions and assumptions.",
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 0_1", "answer of subtask 0_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="subtask_0_2",
        cot_agent_desc=cot_sc_desc_0_2,
        n_repeat=self.max_sc
    )
    logs.append(log_0_2)

    cot_sc_instruction_1_1 = (
        "Sub-task 1_1: Integrate physical principles of gas molecule mean free path and electron scattering effects to analyze how the electron beam modifies collision dynamics and thus the effective mean free path λ2."
    )
    cot_sc_desc_1_1 = {
        'instruction': cot_sc_instruction_1_1,
        'final_decision_instruction': "Sub-task 1_1: Synthesize and choose the most consistent analysis of electron beam effects on mean free path.",
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 0_1", "answer of subtask 0_1", "thinking of subtask 0_2", "answer of subtask 0_2"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="subtask_1_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_reflect_instruction_1_2 = (
        "Sub-task 1_2: Interpret the theoretical or empirical basis for the factor 1.22 in relation to electron scattering cross-sections or collision probabilities, and how it bounds or relates λ2 to λ1."
    )
    critic_instruction_1_2 = (
        "Please review and provide the limitations of provided solutions regarding the factor 1.22 and its impact on the mean free path λ2 compared to λ1."
    )
    cot_reflect_desc_1_2 = {
        'instruction': cot_reflect_instruction_1_2,
        'critic_instruction': critic_instruction_1_2,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 0_2", "answer of subtask 0_2", "thinking of subtask 1_1", "answer of subtask 1_1"]
    }
    results_1_2, log_1_2 = await self.reflexion(
        subtask_id="subtask_1_2",
        reflect_desc=cot_reflect_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Sub-task 2_1: Evaluate the four given choices for the relationship between λ2 and λ1 using the integrated knowledge from previous subtasks, and select the most physically consistent conclusion."
    )
    final_decision_instruction_2_1 = "Sub-task 2_1: Select the most physically consistent conclusion about λ2 relative to λ1 based on all previous analyses."
    debate_desc_2_1 = {
        'instruction': debate_instruction_2_1,
        'final_decision_instruction': final_decision_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer']],
        'context_desc': ["user query", "thinking of subtask 1_1", "answer of subtask 1_1", "thinking of subtask 1_2", "answer of subtask 1_2"],
        'temperature': 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="subtask_2_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])

    return final_answer, logs
