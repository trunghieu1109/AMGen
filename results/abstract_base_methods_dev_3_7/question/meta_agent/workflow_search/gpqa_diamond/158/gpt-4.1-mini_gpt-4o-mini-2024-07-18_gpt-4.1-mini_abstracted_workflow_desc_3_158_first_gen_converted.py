async def forward_158(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Estimate the redshift (z) of the quasar by comparing the observed peak wavelength (~790 nm) to the known rest-frame wavelength of the emission line causing the peak."
    final_decision_instruction_1 = "Sub-task 1: Make final decision on the estimated redshift value."
    debate_desc_1 = {
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "input": [taskInfo],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_1 = {
        "instruction": final_decision_instruction_1,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents_1, final_decision_agent_1, thinking1, answer1, subtask_desc1, list_thinking1, list_answer1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc_1,
        final_decision_desc=final_decision_desc_1,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_1):
            agents.append(f"Debate agent {agent.id}, round {round}, estimating redshift, thinking: {list_thinking1[round][idx].content}; answer: {list_answer1[round][idx].content}")
    agents.append(f"Final Decision agent, estimating redshift, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = "Sub-task 2: Calculate the comoving distance to the quasar using the estimated redshift from subtask_1 and the given Lambda-CDM cosmological parameters (H0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe)."
    cot_agents2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction_2,
        input_list=[taskInfo, thinking1, answer1],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction_2 = "Sub-task 2: Review and refine the comoving distance calculation based on outputs from subtask_1 and SC-CoT results."
    critic_instruction_2 = "Please review the comoving distance calculation and provide any limitations or corrections needed."
    cot_reflect_desc_2 = {
        'instruction': cot_reflect_instruction_2, 'input': [taskInfo, thinking1, answer1, thinking2, answer2], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc_2 = {
        'instruction': critic_instruction_2, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    cot_agent2, critic_agent2, thinking2_reflect, answer2_reflect, subtask_desc2_reflect, list_feedback2, list_correct2, list_thinking2_reflect, list_answer2_reflect = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc_2,
        critic_desc=critic_desc_2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, refining comoving distance calculation, thinking: {thinking2_reflect.content}; answer: {answer2_reflect.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent2.id}, providing feedback, thinking: {list_feedback2[i].content}; answer: {list_correct2[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refining comoving distance calculation, thinking: {list_thinking2_reflect[i].content}; answer: {list_answer2_reflect[i].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_reflect.content}; answer - {answer2_reflect.content}")
    logs.append(subtask_desc2_reflect)

    cot_instruction_3 = "Sub-task 3: Compare the calculated comoving distance from subtask_2 with the provided multiple-choice options (6, 7, 8, 9 Gpc) and select the closest matching choice as the final answer."
    cot_reflect_instruction_3 = "Sub-task 3: Based on the outputs from subtask_2, filter and select the closest matching comoving distance choice from the options."
    critic_instruction_3 = "Please review the selection of the closest comoving distance choice and provide any feedback or corrections."
    cot_agent3, critic_agent3, thinking3, answer3, subtask_desc3, list_feedback3, list_correct3, list_thinking3, list_answer3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc={
            'instruction': cot_reflect_instruction_3, 'input': [taskInfo, thinking1, answer1, thinking2_reflect, answer2_reflect], 'output': ["thinking", "answer"],
            'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
        },
        critic_desc={
            'instruction': critic_instruction_3, 'output': ["feedback", "correct"], 'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, selecting closest comoving distance choice, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent3.id}, providing feedback, thinking: {list_feedback3[i].content}; answer: {list_correct3[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining selection, thinking: {list_thinking3[i].content}; answer: {list_answer3[i].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
