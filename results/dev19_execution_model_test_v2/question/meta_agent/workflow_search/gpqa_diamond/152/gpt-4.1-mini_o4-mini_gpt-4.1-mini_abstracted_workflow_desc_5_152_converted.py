async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_0 = "Sub-task 0: Extract and summarize the defining chemical features, reactants, reagents, and reaction conditions from the query, including the nature of Michael addition and the identities of the reactants and products involved. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_0 = "Sub-task 0: Your problem is to extract and summarize key chemical features and reactants from the query." + debate_instr_0
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0 = self.max_round
    all_thinking_0 = [[] for _ in range(N_max_0)]
    all_answer_0 = [[] for _ in range(N_max_0)]
    subtask_desc0 = {
        "subtask_id": "subtask_0",
        "instruction": debate_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking0, answer0 = await agent([taskInfo], debate_instruction_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_0[r-1] + all_answer_0[r-1]
                thinking0, answer0 = await agent(input_infos_0, debate_instruction_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, extracting and summarizing reactants and features, thinking: {thinking0.content}; answer: {answer0.content}")
            all_thinking_0[r].append(thinking0)
            all_answer_0[r].append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_0[-1] + all_answer_0[-1], "Sub-task 0: Synthesize and choose the most consistent summary of chemical features and reactants." + "Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing summary, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1a = "Sub-task 1a: Based on the output from Sub-task 0, analyze and classify the reactants and reagents in each reaction (A, B, C) based on their chemical properties, roles (Michael donor or acceptor), and expected behavior under the given conditions."
    N_sc_1a = self.max_sc
    cot_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1a)]
    possible_answers_1a = []
    possible_thinkings_1a = []
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_sc_instruction_1a,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1a):
        thinking1a, answer1a = await cot_agents_1a[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1a[i].id}, analyzing reactants and reagents, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a)
        possible_thinkings_1a.append(thinking1a)
    final_decision_agent_1a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1a, answer1a = await final_decision_agent_1a([taskInfo, thinking0, answer0] + possible_thinkings_1a + possible_answers_1a, "Sub-task 1a: Synthesize and choose the most consistent classification of reactants and reagents.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])

    cot_sc_instruction_1b = "Sub-task 1b: Based on the output from Sub-task 0, analyze and classify the possible product structures and intermediates, including tautomeric forms and resonance-stabilized intermediates, based on the reaction mechanisms and conditions."
    N_sc_1b = self.max_sc
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1b)]
    possible_answers_1b = []
    possible_thinkings_1b = []
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_sc_instruction_1b,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1b):
        thinking1b, answer1b = await cot_agents_1b[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, analyzing product structures and intermediates, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b)
        possible_thinkings_1b.append(thinking1b)
    final_decision_agent_1b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1b, answer1b = await final_decision_agent_1b([taskInfo, thinking0, answer0] + possible_thinkings_1b + possible_answers_1b, "Sub-task 1b: Synthesize and choose the most consistent classification of product structures and intermediates.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])

    debate_instr_2 = "Sub-task 2: Transform the classified reactants and intermediates into plausible product structures for reactions A, B, and C, generating variants that reflect different regio- and tautomeric possibilities. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2 = "Sub-task 2: Your problem is to generate plausible product structures based on previous classifications." + debate_instr_2
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_max_2)]
    all_answer_2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1a.content, answer1a.content, thinking1b.content, answer1b.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1a, answer1a, thinking1b, answer1b], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1a, answer1a, thinking1b, answer1b] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, generating plausible product structures, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking_2[r].append(thinking2)
            all_answer_2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1a, answer1a, thinking1b, answer1b] + all_thinking_2[-1] + all_answer_2[-1], "Sub-task 2: Synthesize and choose the most consistent plausible product structures.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing plausible product structures, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = "Sub-task 3: Evaluate and prioritize the generated product variants against the multiple-choice options, selecting the most chemically consistent and plausible products for A, B, and C. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_3 = "Sub-task 3: Your problem is to evaluate and select the best product options." + debate_instr_3
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating and prioritizing product variants, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 3: Synthesize and select the most plausible final products from multiple-choice options.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final products, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
