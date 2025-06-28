async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1a: Define and explain the thermodynamic concept of oxygen as an oxidant in aqueous solutions, focusing on standard reduction potentials and how they differ between acidic and basic media."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, defining thermodynamic concept of oxygen oxidant, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = "Sub-task 1b: Gather and tabulate empirical thermodynamic data (standard electrode potentials) for oxygen reduction reactions in acidic and basic solutions to establish oxygen's relative oxidizing strength in each medium. Include tabulated data referencing standard potentials."
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, gathering empirical thermodynamic data, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_2a = "Sub-task 2a: Define the kinetic factors influencing oxygen reduction reaction (ORR) rates in acidic and basic solutions, including mechanistic pathways, activation energies, overpotentials, and rate constants."
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_2a, answer_2a = await cot_agent_2a([taskInfo], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, defining kinetic factors of ORR, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    debate_instruction_2b = "Sub-task 2b: Collect and summarize empirical kinetic data for ORR in acidic and basic media, including exchange current densities, overpotentials, and known rate-determining steps, to characterize reaction speed differences. Two agents will debate the hypothesis: Agent 1 argues ORR is faster in acidic media; Agent 2 argues ORR is faster in basic media. Each must cite empirical data and mechanistic insights."
    debate_agents_2b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2b = self.max_round
    all_thinking_2b = [[] for _ in range(N_max_2b)]
    all_answer_2b = [[] for _ in range(N_max_2b)]
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": debate_instruction_2b,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2b):
        for i, agent in enumerate(debate_agents_2b):
            if r == 0:
                thinking_2b, answer_2b = await agent([taskInfo, thinking_2a, answer_2a], debate_instruction_2b, r, is_sub_task=True)
            else:
                input_infos_2b = [taskInfo, thinking_2a, answer_2a] + all_thinking_2b[r-1] + all_answer_2b[r-1]
                thinking_2b, answer_2b = await agent(input_infos_2b, debate_instruction_2b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating ORR kinetics, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
            all_thinking_2b[r].append(thinking_2b)
            all_answer_2b[r].append(answer_2b)
    final_decision_agent_2b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2b, answer_2b = await final_decision_agent_2b([taskInfo] + all_thinking_2b[-1] + all_answer_2b[-1], "Sub-task 2b: Adjudicate the debate and conclude the kinetic behavior of oxygen reduction in acidic vs basic media based on empirical evidence and mechanistic insights.", is_sub_task=True)
    agents.append(f"Final Decision agent, adjudicating ORR kinetics, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_instruction_3a = "Sub-task 3a: Analyze the thermodynamic data from Sub-task 1b to determine whether oxygen is a stronger or weaker oxidant in basic solutions compared to acidic solutions."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "CoT"
    }
    thinking_3a, answer_3a = await cot_agent_3a([taskInfo, thinking_1b, answer_1b], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, analyzing thermodynamic data, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_instruction_3b = "Sub-task 3b: Analyze the kinetic data from Sub-task 2b to determine whether oxygen reduction proceeds faster or slower in acidic solutions compared to basic solutions, incorporating mechanistic insights and empirical evidence."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "CoT"
    }
    thinking_3b, answer_3b = await cot_agent_3b([taskInfo, thinking_2b, answer_2b], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, analyzing kinetic data, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_reflect_instruction_4 = "Sub-task 4: Conduct a critical reflexion and cross-validation step to reconcile thermodynamic and kinetic analyses from Sub-tasks 3a and 3b, challenge assumptions, and ensure consistency and accuracy of conclusions regarding oxygen's oxidant strength and reaction speed in acidic and basic media."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking_3a, answer_3a, thinking_3b, answer_3b]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask_3a", "answer of subtask_3a", "thinking of subtask_3b", "answer of subtask_3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking_4, answer_4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, reconciling thermodynamic and kinetic analyses, thinking: {thinking_4.content}; answer: {answer_4.content}")
    for i in range(N_max_4):
        feedback_4, correct_4 = await critic_agent_4([taskInfo, thinking_4, answer_4], "Please review the reconciliation of thermodynamic and kinetic analyses and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback_4.content}; answer: {correct_4.content}")
        if correct_4.content == "True":
            break
        cot_inputs_4.extend([thinking_4, answer_4, feedback_4])
        thinking_4, answer_4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining reconciliation, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Combine the validated thermodynamic and kinetic conclusions from Sub-task 4 to identify the correct combination of descriptors (weaker/stronger and faster/slower) that accurately describes oxygen's behavior in the given conditions."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4, answer_4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4, answer_4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, combining conclusions, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Make final decision on the correct combination of descriptors.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct combination, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    
    debate_instruction_6 = "Sub-task 6: Map the identified correct combination from Sub-task 5 to the provided multiple-choice options and select the corresponding letter choice (A, B, C, or D)."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking_6 = [[] for _ in range(N_max_6)]
    all_answer_6 = [[] for _ in range(N_max_6)]
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask_5", "answer of subtask_5"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking_6, answer_6 = await agent([taskInfo, thinking_5, answer_5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking_5, answer_5] + all_thinking_6[r-1] + all_answer_6[r-1]
                thinking_6, answer_6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, mapping to multiple-choice options, thinking: {thinking_6.content}; answer: {answer_6.content}")
            all_thinking_6[r].append(thinking_6)
            all_answer_6[r].append(answer_6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_6, answer_6 = await final_decision_agent_6([taskInfo] + all_thinking_6[-1] + all_answer_6[-1], "Sub-task 6: Make final decision on the correct multiple-choice letter.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting multiple-choice letter, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_6, answer_6, sub_tasks, agents)
    return final_answer, logs