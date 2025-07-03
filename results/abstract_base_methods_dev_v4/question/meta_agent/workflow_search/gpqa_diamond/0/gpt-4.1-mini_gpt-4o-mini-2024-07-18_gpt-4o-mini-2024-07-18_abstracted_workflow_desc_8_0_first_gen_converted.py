async def forward_0(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction_1 = "Sub-task 1: Calculate the energy uncertainty (ΔE) for each quantum state using their given lifetimes (τ) applying the energy-time uncertainty relation ΔE ≈ ħ / τ, with ħ = 6.582119569e-16 eV·s. Use the lifetimes 10^-9 s and 10^-8 s respectively."
    cot_agents1, thinking1, answer1, subtask_desc1, list_thinking1, list_answer1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_sc_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    cot_reflect_instruction_1 = "Sub-task 1 Reflexion: Review the calculated energy uncertainties for both states and verify correctness and consistency."
    critic_instruction_1 = "Please review the energy uncertainty calculations and provide feedback on any errors or improvements needed."
    cot_reflect_desc_1 = {
        'instruction': cot_reflect_instruction_1,
        'input': [taskInfo, thinking1, answer1],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    cot_agent1_reflect, critic_agent1, thinking1_reflect, answer1_reflect, subtask_desc1_reflect, list_feedback1, list_correct1, list_thinking1_reflect, list_answer1_reflect = await self.reflexion(
        subtask_id="subtask_1_reflect",
        cot_reflect_desc=cot_reflect_desc_1,
        critic_desc=critic_desc_1,
        n_repeat=self.max_round
    )
    agents.append(f"SC_CoT agents {[agent.id for agent in cot_agents1]}, calculated energy uncertainties, thinking: {thinking1.content}; answer: {answer1.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent1.id}, feedback round {i}, thinking: {list_feedback1[i].content}; answer: {list_correct1[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent1_reflect.id}, refining calculations round {i}, thinking: {list_thinking1_reflect[i+1].content}; answer: {list_answer1_reflect[i+1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_reflect.content}; answer - {answer1_reflect.content}")
    logs.append(subtask_desc1)
    logs.append(subtask_desc1_reflect)
    
    cot_instruction_2 = "Sub-task 2: Determine the minimum energy difference required to clearly distinguish the two energy levels based on the larger of the two energy uncertainties calculated in subtask_1."
    cot_agents2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_instruction_2,
        input_list=[taskInfo, thinking1_reflect, answer1_reflect],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    agents.append(f"CoT-SC agents {[agent.id for agent in cot_agents2]}, determined minimum energy difference, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append(subtask_desc2)
    
    debate_instruction_3 = "Sub-task 3: Compare the given multiple-choice energy difference options with the minimum required energy difference from subtask_2 to identify which option allows clear resolution of the two states."
    final_decision_instruction_3 = "Sub-task 3: Make final decision on which energy difference option allows clear resolution of the two quantum states."
    debate_desc_3 = {
        "instruction": debate_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "input": [taskInfo, thinking2, answer2],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_3 = {
        "instruction": final_decision_instruction_3,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents3, final_decision_agent3, thinking3, answer3, subtask_desc3, list_thinking3, list_answer3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc_3,
        final_decision_desc=final_decision_desc_3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents3):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing options and reasoning, thinking: {list_thinking3[round][idx].content}; answer: {list_answer3[round][idx].content}")
    agents.append(f"Final Decision agent {final_decision_agent3.id}, finalizing choice, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs