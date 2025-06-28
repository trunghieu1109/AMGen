async def forward_183(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Identify and clearly label all relevant structural features, substituents, and functional groups of the target molecule 2-(tert-butyl)-1-ethoxy-3-nitrobenzene, including the exact positions on the benzene ring, to establish the synthetic goals starting from benzene."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying target molecule features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    debate_instruction_2a = "Sub-task 2a: Number and label all positions on the benzene ring after each substitution step (tert-butyl, ethoxy, nitro) to avoid ambiguity in regioselectivity analysis, based on Sub-task 1 outputs."
    debate_agents_2a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2a = self.max_round
    all_thinking2a = [[] for _ in range(N_max_2a)]
    all_answer2a = [[] for _ in range(N_max_2a)]
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": debate_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2a):
        for i, agent in enumerate(debate_agents_2a):
            if r == 0:
                thinking2a, answer2a = await agent([taskInfo, thinking1, answer1], debate_instruction_2a, r, is_sub_task=True)
            else:
                input_infos_2a = [taskInfo, thinking1, answer1] + all_thinking2a[r-1] + all_answer2a[r-1]
                thinking2a, answer2a = await agent(input_infos_2a, debate_instruction_2a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, numbering and labeling benzene positions, thinking: {thinking2a.content}; answer: {answer2a.content}")
            all_thinking2a[r].append(thinking2a)
            all_answer2a[r].append(answer2a)
    final_decision_agent_2a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2a, answer2a = await final_decision_agent_2a([taskInfo] + all_thinking2a[-1] + all_answer2a[-1], "Sub-task 2a: Make final decision on numbering and labeling of benzene ring positions after substitutions.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting numbering scheme, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    
    debate_instruction_2b = "Sub-task 2b: Systematically analyze the directing effects of each substituent (tert-butyl, ethoxy, nitro) on the benzene ring, listing all possible nitration sites and assessing their feasibility based on electronic and steric factors, using the numbering from Sub-task 2a."
    debate_agents_2b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2b = self.max_round
    all_thinking2b = [[] for _ in range(N_max_2b)]
    all_answer2b = [[] for _ in range(N_max_2b)]
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": debate_instruction_2b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2b):
        for i, agent in enumerate(debate_agents_2b):
            if r == 0:
                thinking2b, answer2b = await agent([taskInfo, thinking1, answer1, thinking2a, answer2a], debate_instruction_2b, r, is_sub_task=True)
            else:
                input_infos_2b = [taskInfo, thinking1, answer1, thinking2a, answer2a] + all_thinking2b[r-1] + all_answer2b[r-1]
                thinking2b, answer2b = await agent(input_infos_2b, debate_instruction_2b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing directing effects and nitration sites, thinking: {thinking2b.content}; answer: {answer2b.content}")
            all_thinking2b[r].append(thinking2b)
            all_answer2b[r].append(answer2b)
    final_decision_agent_2b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2b, answer2b = await final_decision_agent_2b([taskInfo, thinking1, answer1, thinking2a, answer2a] + all_thinking2b[-1] + all_answer2b[-1], "Sub-task 2b: Make final decision on directing effects and feasible nitration sites.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting directing effects analysis, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {
        "thinking": thinking2b,
        "answer": answer2b
    }
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_reflect_instruction_3a = "Sub-task 3a: Map each reagent and reaction condition from the given options to their corresponding chemical transformations (e.g., Friedel-Crafts alkylation, nitration, sulfonation, reduction, diazotization, substitution, hydrolysis), ensuring clarity on the role of each step, based on Sub-task 2b outputs."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3a = self.max_round
    cot_inputs_3a = [taskInfo, thinking2b, answer2b]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, mapping reagents to transformations, thinking: {thinking3a.content}; answer: {answer3a.content}")
    for i in range(N_max_3a):
        feedback, correct = await critic_agent_3a([taskInfo, thinking3a, answer3a], "Please review the reagent-to-transformation mapping and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3a.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3a.extend([thinking3a, answer3a, feedback])
        thinking3a, answer3a = await cot_agent_3a(cot_inputs_3a, cot_reflect_instruction_3a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining reagent mapping, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_reflect_instruction_3b = "Sub-task 3b: Perform a detailed synthetic feasibility analysis of the mapped transformations, explicitly considering protection/deprotection strategies, blocking groups (such as sulfonation/desulfonation), regiochemical control, and practical challenges to achieve high-yield synthesis, based on Sub-task 3a outputs."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking3a, answer3a]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, analyzing synthetic feasibility, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max_3b):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "Critically evaluate the synthetic feasibility analysis, focusing on protection/deprotection, blocking groups, and regiochemical control.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining synthetic feasibility analysis, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    debate_instruction_4 = "Sub-task 4: Using a debate approach, evaluate multiple plausible sequences of reactions starting from benzene, incorporating directing effects, blocking group strategies, and synthetic feasibility to propose the most chemically sound and high-yielding synthetic route to 2-(tert-butyl)-1-ethoxy-3-nitrobenzene, based on Sub-task 3b outputs."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3b, answer3b], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3b, answer3b] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating reaction sequences, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the most feasible and high-yielding synthetic route.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting feasible synthetic route, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Compare the proposed feasible synthetic route against each provided multiple-choice sequence (choices 1 to 4), critically assessing their alignment with the ideal route and identifying the sequence that would lead to high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene, based on Sub-task 4 outputs."
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing sequences, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 6: Select the correct multiple-choice answer (A, B, C, or D) corresponding to the sequence that best fits the high-yield synthesis pathway determined.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct sequence, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs