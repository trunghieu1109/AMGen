async def forward_164(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = (
        "Sub-task 1: Analyze the polymerization context: understand ethylene polymerization with a homogeneous organometallic catalyst producing high-density polyethylene, "
        "and the goal to introduce regular branches using a second catalyst system with only ethylene as monomer, clarifying the chemical and industrial constraints."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing polymerization context, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = (
        "Sub-task 2: Investigate the industrial implementation status of dual catalyst systems that produce branched polyethylene using only ethylene (no comonomers), "
        "including known commercial examples such as the Phillips Cr catalyst and Brookhart Ni diimine systems, and their geographic deployment, to verify the validity of statement A."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, investigating industrial implementation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = (
        "Sub-task 3: Evaluate the role and effectiveness of aluminum-based activators in the essential additional reaction step for introducing regular branches in polyethylene using dual catalyst systems with only ethylene, "
        "referencing known catalyst-activator combinations such as Ziegler-Natta and metallocene systems, and their mechanistic requirements to assess statement B."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, evaluating aluminum-based activators, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the evaluation of aluminum-based activators and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining evaluation of aluminum-based activators, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4 = (
        "Sub-task 4: Assess the catalytic activity and feasibility of group VIa transition metal catalysts combined with specific activators for producing branched polyethylene from ethylene alone, "
        "including mechanistic insights and industrial relevance, referencing catalysts such as chromium-based Phillips catalysts, to evaluate statement C."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, assessing group VIa catalysts, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = (
        "Sub-task 5: Examine the use of certain noble metal catalysts for ethylene polymerization to produce branched polymers, focusing on their catalytic performance, cost factors, and industrial practicality, "
        "to evaluate statement D through a debate between agents arguing for and against their industrial use."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating noble metal catalysts, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the cost and practicality of noble metal catalysts.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on noble metal catalysts, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_sc_instruction_6a = (
        "Sub-task 6a: Identify and analyze contradictions or consistencies among the four statements (A–D) based on the findings from subtasks 2 to 5, "
        "highlighting any conflicts or corroborations with established industrial and chemical knowledge."
    )
    cot_agents_6a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6a = []
    thinkingmapping_6a = {}
    answermapping_6a = {}
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_sc_instruction_6a,
        "context": ["user query", "thinking and answers of subtasks 2,3,4,5", "answers of subtasks 2,3,4,5"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6a, answer6a = await cot_agents_6a[i]([
            taskInfo, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5
        ], cot_sc_instruction_6a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6a[i].id}, analyzing contradictions among statements, thinking: {thinking6a.content}; answer: {answer6a.content}")
        possible_answers_6a.append(answer6a.content)
        thinkingmapping_6a[answer6a.content] = thinking6a
        answermapping_6a[answer6a.content] = answer6a
    answer6a_content = Counter(possible_answers_6a).most_common(1)[0][0]
    thinking6a = thinkingmapping_6a[answer6a_content]
    answer6a = answermapping_6a[answer6a_content]
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {
        "thinking": thinking6a,
        "answer": answer6a
    }
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    
    cot_sc_instruction_6b = (
        "Sub-task 6b: Select the single correct statement regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system, "
        "based on the critical analysis and reconciliation of evidence from previous subtasks, ensuring the choice is supported by industrial examples and catalytic chemistry."
    )
    cot_agents_6b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6b = []
    thinkingmapping_6b = {}
    answermapping_6b = {}
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_sc_instruction_6b,
        "context": ["user query", "thinking and answer of subtask 6a", "answer of subtask 6a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6b, answer6b = await cot_agents_6b[i]([taskInfo, thinking6a, answer6a], cot_sc_instruction_6b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6b[i].id}, selecting the correct statement, thinking: {thinking6b.content}; answer: {answer6b.content}")
        possible_answers_6b.append(answer6b.content)
        thinkingmapping_6b[answer6b.content] = thinking6b
        answermapping_6b[answer6b.content] = answer6b
    answer6b_content = Counter(possible_answers_6b).most_common(1)[0][0]
    thinking6b = thinkingmapping_6b[answer6b_content]
    answer6b = answermapping_6b[answer6b_content]
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {
        "thinking": thinking6b,
        "answer": answer6b
    }
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6b, answer6b, sub_tasks, agents)
    return final_answer, logs
