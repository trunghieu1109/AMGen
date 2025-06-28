async def forward_26(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction_1 = "Sub-task 1: Identify and explicitly recognize the ribonucleoprotein particle in the dialogue as the Signal Recognition Particle (SRP), detailing its biological role in binding the nascent polypeptide chain at the ER-bound ribosome."
    N1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, identifying ribonucleoprotein particle as SRP, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    answer1_content = Counter(possible_answers_1).most_common(1)[0][0]
    thinking1 = thinkingmapping_1[answer1_content]
    answer1 = answermapping_1[answer1_content]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_reflect_instruction_2 = "Sub-task 2: Interpret the dialogues contextual clues—such as 'Pause there for a minute' and 'you really need some sugar'—to understand the biological significance of the interaction, specifically the SRP-mediated pause in translation and the initiation of co-translational translocation and glycosylation."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, interpreting biological significance of SRP pause and glycosylation, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_2):
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], "please review the interpretation of SRP-mediated pause and glycosylation initiation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining interpretation of SRP pause and glycosylation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    debate_instruction_3_4 = "Sub-task 3 and 4: Determine the precise cellular location where the SRP and nascent chain meet, emphasizing the ER-bound ribosome and rough ER context, and identify the immediate and subsequent destinations of the nascent chain after SRP interaction, distinguishing initial translocation into the rough ER and eventual trafficking to Golgi and beyond. Use biological knowledge to ensure plausibility."
    debate_agents_3_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_4 = self.max_round
    all_thinking_3_4 = [[] for _ in range(N_max_3_4)]
    all_answer_3_4 = [[] for _ in range(N_max_3_4)]
    subtask_desc3_4 = {
        "subtask_id": "subtask_3_4",
        "instruction": debate_instruction_3_4,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_4):
        for i, agent in enumerate(debate_agents_3_4):
            if r == 0:
                thinking3_4, answer3_4 = await agent([taskInfo, thinking2, answer2], debate_instruction_3_4, r, is_sub_task=True)
            else:
                input_infos_3_4 = [taskInfo, thinking2, answer2] + all_thinking_3_4[r-1] + all_answer_3_4[r-1]
                thinking3_4, answer3_4 = await agent(input_infos_3_4, debate_instruction_3_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determining meeting location and nascent chain destinations, thinking: {thinking3_4.content}; answer: {answer3_4.content}")
            all_thinking_3_4[r].append(thinking3_4)
            all_answer_3_4[r].append(answer3_4)
    final_decision_agent_3_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_4, answer3_4 = await final_decision_agent_3_4([taskInfo] + all_thinking_3_4[-1] + all_answer_3_4[-1], "Sub-task 3 and 4: Make final decision on the meeting location and nascent chain destinations.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting meeting location and nascent chain destinations, thinking: {thinking3_4.content}; answer: {answer3_4.content}")
    sub_tasks.append(f"Sub-task 3 and 4 output: thinking - {thinking3_4.content}; answer - {answer3_4.content}")
    subtask_desc3_4['response'] = {
        "thinking": thinking3_4,
        "answer": answer3_4
    }
    logs.append(subtask_desc3_4)
    print("Step 3 and 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Validate and cross-check the identified meeting location and nascent chain destination against canonical protein trafficking pathways and the provided multiple-choice options, selecting the biologically accurate answer that best fits the dialogue context."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 3 and 4", "answer of subtask 3 and 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3_4, answer3_4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3_4, answer3_4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating and selecting final answer, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs