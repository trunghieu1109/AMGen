async def forward_164(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Identify and understand the characteristics of the initial polymerization system: a homogeneous organometallic catalyst producing high-density polyethylene from ethylene."
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
    cot_instruction_2a = "Sub-task 2a: Review the chain-walking mechanism of Ni(II)/Pd(II) α-diimine catalysts in ethylene polymerization, focusing on how branching occurs."
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking2a, answer2a = await cot_agent_2a([taskInfo, thinking1, answer1], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, reviewing chain-walking mechanism, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    cot_instruction_2b = "Sub-task 2b: Evaluate whether the chain-walking mechanism produces regular (uniformly spaced) branches or random branches in the polymer backbone. Explicitly define 'regular' as uniformly spaced branches and 'random' as branches with variable spacing and distribution. Justify your evaluation based on polymer microstructure."
    N_2b = self.max_sc
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_2b)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_2b):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, evaluating branch regularity, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    debate_instruction_2c = "Sub-task 2c: Conduct a structured debate between two perspectives: one arguing that Ni/Pd chain-walking catalysts produce random branches, and another arguing they can be tuned to produce regular branches. Summarize the consensus or divergence and provide a reasoned conclusion."
    debate_agents_2c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2c = self.max_round
    all_thinking_2c = [[] for _ in range(N_max_2c)]
    all_answer_2c = [[] for _ in range(N_max_2c)]
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": debate_instruction_2c,
        "context": ["user query", "thinking and answer of subtask 2b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2c):
        for i, agent in enumerate(debate_agents_2c):
            if r == 0:
                input_infos_2c = [taskInfo, thinking2b, answer2b]
            else:
                input_infos_2c = [taskInfo, thinking2b, answer2b] + all_thinking_2c[r-1] + all_answer_2c[r-1]
            thinking2c, answer2c = await agent(input_infos_2c, debate_instruction_2c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating branch regularity, thinking: {thinking2c.content}; answer: {answer2c.content}")
            all_thinking_2c[r].append(thinking2c)
            all_answer_2c[r].append(answer2c)
    final_decision_agent_2c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2c, answer2c = await final_decision_agent_2c([taskInfo] + all_thinking_2c[-1] + all_answer_2c[-1], "Sub-task 2c: Summarize the debate on branch regularity and provide a reasoned conclusion.", is_sub_task=True)
    agents.append(f"Final Decision agent 2c, synthesizing debate results on branch regularity, thinking: {thinking2c.content}; answer: {answer2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {"thinking": thinking2c, "answer": answer2c}
    logs.append(subtask_desc2c)
    print("Step 2c: ", sub_tasks[-1])
    cot_instruction_2d = "Sub-task 2d: Perform a domain-knowledge check using curated literature or expert input to verify the microstructure of polymers produced by Ni/Pd chain-walking catalysts and confirm the nature of branching (regular vs random)."
    cot_agent_2d = LLMAgentBase(["thinking", "answer"], "Chemistry Expert Agent", model=self.node_model, temperature=0.0)
    subtask_desc2d = {
        "subtask_id": "subtask_2d",
        "instruction": cot_instruction_2d,
        "context": ["user query", "thinking and answer of subtask 2c"],
        "agent_collaboration": "Expert Check"
    }
    thinking2d, answer2d = await cot_agent_2d([taskInfo, thinking2c, answer2c], cot_instruction_2d, is_sub_task=True)
    agents.append(f"Chemistry Expert agent {cot_agent_2d.id}, verifying polymer microstructure and branching nature, thinking: {thinking2d.content}; answer: {answer2d.content}")
    sub_tasks.append(f"Sub-task 2d output: thinking - {thinking2d.content}; answer - {answer2d.content}")
    subtask_desc2d['response'] = {"thinking": thinking2d, "answer": answer2d}
    logs.append(subtask_desc2d)
    print("Step 2d: ", sub_tasks[-1])
    cot_reflect_instruction_3 = "Sub-task 3: Evaluate the validity of the statement that such combined dual catalyst systems are already implemented on an industrial scale in the US, considering the verified branching mechanism from Sub-task 2d."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking2d, answer2d]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, evaluating industrial implementation statement, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3], "please review the validity of the industrial scale implementation statement and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining evaluation of industrial implementation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking and answer of subtask 2d"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_reflect_instruction_4 = "Sub-task 4: Assess the role and effectiveness of aluminum-based activators in the essential additional reaction step for introducing regular branches in the polymer backbone, based on the confirmed catalyst mechanisms from Sub-task 2d."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking2d, answer2d]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, assessing aluminum-based activators, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4], "please review the assessment of aluminum-based activators and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining assessment of aluminum-based activators, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking and answer of subtask 2d"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_reflect_instruction_5 = "Sub-task 5: Determine whether catalysts from group VIa transition metals combined with specific activators can be used to achieve the desired polymer branching with ethylene only, incorporating the verified branching regularity information from Sub-task 2d."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking2d, answer2d]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, determining group VIa catalyst usage, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback5, correct5 = await critic_agent_5([taskInfo, thinking5, answer5], "please review the determination of group VIa catalyst usage and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback5.content}; answer: {correct5.content}")
        if correct5.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining determination of group VIa catalyst usage, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking and answer of subtask 2d"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_reflect_instruction_6 = "Sub-task 6: Analyze the feasibility and cost implications of using certain noble metal catalysts for the branching reaction in ethylene polymerization, considering their ability to produce regular branches and industrial applicability, based on Sub-task 2d."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking2d, answer2d]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, analyzing noble metal catalysts feasibility and cost, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6, answer6], "please review the analysis of noble metal catalysts feasibility and cost and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining analysis of noble metal catalysts, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking and answer of subtask 2d"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction_7 = "Sub-task 7: Integrate the evaluations from subtasks 3 to 6 to identify all correct statements among the four given options regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system. Allow for multiple correct answers and justify each."
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_7,
        "context": ["user query", "thinking and answer of subtask 3", "thinking and answer of subtask 4", "thinking and answer of subtask 5", "thinking and answer of subtask 6"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                input_infos_7 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6]
            else:
                input_infos_7 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
            thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating evaluations and deciding correct statements, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on all correct statements regarding polymer branching with ethylene and dual catalyst system, allowing multiple correct answers.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing debate results, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_reflect_instruction_8 = "Sub-task 8: Perform a final validation step to cross-check the integrated answer against the question’s requirements, ensuring that multiple correct statements are identified if applicable and that the reasoning aligns with domain knowledge."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking7, answer7]
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, performing final validation, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback8, correct8 = await critic_agent_8([taskInfo, thinking8, answer8], "please review the final integrated answer for completeness and correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback8.content}; answer: {correct8.content}")
        if correct8.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback8])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining final validation, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", "thinking and answer of subtask 7"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs