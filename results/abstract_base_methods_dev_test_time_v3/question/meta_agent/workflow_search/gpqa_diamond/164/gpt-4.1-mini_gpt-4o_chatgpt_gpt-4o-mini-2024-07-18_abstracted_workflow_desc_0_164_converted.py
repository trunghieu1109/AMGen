async def forward_164(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify and understand the characteristics of the initial polymerization system: a homogeneous organometallic catalyst producing high-density polyethylene from ethylene, including the nature of the polymer and catalyst type."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying initial polymerization system, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Analyze the requirement and chemical feasibility of introducing a second catalyst system to create regular branches in the polyethylene backbone using only ethylene as the monomer, clarifying the polymer structure implications, based on Sub-task 1 output."
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing second catalyst system requirement, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    domain_knowledge_instruction_3 = "Sub-task 3: Retrieve and review authoritative domain knowledge and industrial examples on dual catalyst systems for ethylene polymerization that produce branched polymers, focusing on the existence and industrial implementation of such systems in the US, based on Sub-task 2 output."
    domain_knowledge_instruction_4 = "Sub-task 4: Critically evaluate the statement regarding the inefficacy of aluminum-based activators in the essential additional reaction step for branching, by consulting chemical literature and catalyst activation mechanisms relevant to group VIa transition metal catalysts and branching reactions, based on Sub-task 2 output."
    domain_knowledge_instruction_5 = "Sub-task 5: Assess the validity of using group VIa transition metal catalysts combined with specific activators for producing branched polyethylene from ethylene only, supported by chemical data and known catalyst systems, based on Sub-task 2 output."
    domain_knowledge_instruction_6 = "Sub-task 6: Evaluate the feasibility and economic considerations of using noble metal catalysts for the branching reaction in ethylene polymerization, including cost analysis and industrial practicality, based on Sub-task 2 output."
    
    debate_roles = self.debate_role
    N_max_debate = self.max_round
    
    def create_debate_agents():
        return [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": domain_knowledge_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": domain_knowledge_instruction_4,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": domain_knowledge_instruction_5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": domain_knowledge_instruction_6,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    
    debate_agents_3 = create_debate_agents()
    debate_agents_4 = create_debate_agents()
    debate_agents_5 = create_debate_agents()
    debate_agents_6 = create_debate_agents()
    
    all_thinking_3 = [[] for _ in range(N_max_debate)]
    all_answer_3 = [[] for _ in range(N_max_debate)]
    all_thinking_4 = [[] for _ in range(N_max_debate)]
    all_answer_4 = [[] for _ in range(N_max_debate)]
    all_thinking_5 = [[] for _ in range(N_max_debate)]
    all_answer_5 = [[] for _ in range(N_max_debate)]
    all_thinking_6 = [[] for _ in range(N_max_debate)]
    all_answer_6 = [[] for _ in range(N_max_debate)]
    
    for r in range(N_max_debate):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], domain_knowledge_instruction_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2, answer2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking, answer = await agent(input_infos, domain_knowledge_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating industrial implementation statement, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3[r].append(thinking)
            all_answer_3[r].append(answer)
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], domain_knowledge_instruction_4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2, answer2] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking, answer = await agent(input_infos, domain_knowledge_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating aluminum-based activators statement, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_4[r].append(thinking)
            all_answer_4[r].append(answer)
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], domain_knowledge_instruction_5, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2, answer2] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking, answer = await agent(input_infos, domain_knowledge_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating group VIa catalyst statement, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_5[r].append(thinking)
            all_answer_5[r].append(answer)
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], domain_knowledge_instruction_6, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2, answer2] + all_thinking_6[r-1] + all_answer_6[r-1]
                thinking, answer = await agent(input_infos, domain_knowledge_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating noble metal catalyst statement, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_6[r].append(thinking)
            all_answer_6[r].append(answer)
    
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 3: Make final decision on the industrial implementation statement.", is_sub_task=True)
    agents.append(f"Final Decision agent, industrial implementation statement, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Make final decision on the aluminum-based activators statement.", is_sub_task=True)
    agents.append(f"Final Decision agent, aluminum-based activators statement, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Make final decision on the group VIa catalyst statement.", is_sub_task=True)
    agents.append(f"Final Decision agent, group VIa catalyst statement, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking_6[-1] + all_answer_6[-1], "Sub-task 6: Make final decision on the noble metal catalyst statement.", is_sub_task=True)
    agents.append(f"Final Decision agent, noble metal catalyst statement, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    debate_instruction_7 = "Sub-task 7: Conduct a structured debate-style critical examination of each of the four statements, with supporting and opposing arguments based on the retrieved domain knowledge and evaluations from Sub-tasks 3 to 6, ensuring contradictions and consistencies are identified."
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_7 = [[] for _ in range(self.max_round)]
    all_answer_7 = [[] for _ in range(self.max_round)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking and answer of subtask 3", "thinking and answer of subtask 4", "thinking and answer of subtask 5", "thinking and answer of subtask 6"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6] + all_thinking_7[r-1] + all_answer_7[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating integrated statements, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_7[r].append(thinking)
            all_answer_7[r].append(answer)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_8([taskInfo] + all_thinking_7[-1] + all_answer_7[-1], "Sub-task 8: Integrate the outcomes of the debate and evaluations to identify the single correct statement regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system, explicitly justifying the choice and excluding others.", is_sub_task=True)
    agents.append(f"Final Decision agent, integrating debate outcomes, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs