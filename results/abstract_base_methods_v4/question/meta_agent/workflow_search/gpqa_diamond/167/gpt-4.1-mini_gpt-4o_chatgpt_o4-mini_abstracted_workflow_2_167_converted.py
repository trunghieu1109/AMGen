async def forward_167(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = (
        "Sub-task 1: Define and clarify each listed issue in genomics data analysis: mutually incompatible data formats, the 'chr' / 'no chr' confusion, "
        "reference assembly mismatch, and incorrect ID conversion. Establish a precise understanding of their nature, typical manifestations, "
        "and how they affect data processing.")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, define and clarify genomics data issues, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = (
        "Sub-task 2: Identify and describe common scenarios or workflows in genomics data analysis where each issue typically arises, "
        "emphasizing how these issues manifest in practice and their potential to cause errors, based on clarifications from Sub-task 1.")
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identify common scenarios for genomics issues, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    debate_instruction_3a = (
        "Sub-task 3a: Classify each issue based on typical error detectability mode: distinguish between issues that cause explicit failures or warnings "
        "(e.g., pipeline aborts) versus those that silently propagate erroneous results without immediate detection. Use domain evidence and examples to support classification.")
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3a = self.max_round
    all_thinking3a = [[] for _ in range(N_max_3a)]
    all_answer3a = [[] for _ in range(N_max_3a)]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": debate_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3a):
        for i, agent in enumerate(debate_agents_3a):
            if r == 0:
                thinking3a, answer3a = await agent([taskInfo, thinking2, answer2], debate_instruction_3a, r, is_sub_task=True)
            else:
                input_infos_3a = [taskInfo, thinking2, answer2] + all_thinking3a[r-1] + all_answer3a[r-1]
                thinking3a, answer3a = await agent(input_infos_3a, debate_instruction_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, classifying error detectability, thinking: {thinking3a.content}; answer: {answer3a.content}")
            all_thinking3a[r].append(thinking3a)
            all_answer3a[r].append(answer3a)
    thinking3a_final = all_thinking3a[-1][0]
    answer3a_final = all_answer3a[-1][0]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a_final.content}; answer - {answer3a_final.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a_final, "answer": answer3a_final}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_reflect_instruction_3b = (
        "Sub-task 3b: Estimate the real-world frequency and impact of each issue causing difficult-to-spot erroneous results in genomics data analysis, "
        "using domain knowledge, empirical evidence, or documented case studies to support the assessment, based on Sub-task 3a outputs.")
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking2, answer2, thinking3a_final, answer3a_final]
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, estimating frequency and impact, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max_3b):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b],
                                                 "Please review the frequency and impact estimation of genomics issues causing difficult-to-spot errors and provide limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining frequency and impact estimation, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Reflexion",
        "response": {"thinking": thinking3b, "answer": answer3b}
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_sc_instruction_4 = (
        "Sub-task 4: Synthesize the classification from Sub-task 3a and frequency assessments from Sub-task 3b to rank the issues by their commonality and subtlety "
        "as sources of difficult-to-spot erroneous results, prioritizing those that silently propagate errors and occur frequently.")
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3a_final, answer3a_final, thinking3b, answer3b], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, synthesize and rank genomics issues, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5a = (
        "Sub-task 5a: Map the ranked issues from Sub-task 4 to the provided multiple-choice options, identifying which choices correspond to the highest-ranked issues "
        "as sources of difficult-to-spot errors.")
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": debate_instruction_5a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking4, answer4], debate_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, mapping ranked issues to choices, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    debate_instruction_5b = (
        "Sub-task 5b: Critically evaluate and select the best multiple-choice answer that aligns with the synthesized ranking and domain evidence, "
        "ensuring consistency and prioritizing domain validity over majority voting or overly broad selections.")
    debate_agents_5b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5b = self.max_round
    all_thinking5b = [[] for _ in range(N_max_5b)]
    all_answer5b = [[] for _ in range(N_max_5b)]
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": debate_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5b):
        for i, agent in enumerate(debate_agents_5b):
            if r == 0:
                thinking5b, answer5b = await agent([taskInfo, thinking5a, answer5a], debate_instruction_5b, r, is_sub_task=True)
            else:
                input_infos_5b = [taskInfo, thinking5a, answer5a] + all_thinking5b[r-1] + all_answer5b[r-1]
                thinking5b, answer5b = await agent(input_infos_5b, debate_instruction_5b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating and selecting best answer, thinking: {thinking5b.content}; answer: {answer5b.content}")
            all_thinking5b[r].append(thinking5b)
            all_answer5b[r].append(answer5b)
    final_decision_agent_5b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5b_final, answer5b_final = await final_decision_agent_5b([taskInfo] + all_thinking5b[-1] + all_answer5b[-1],
                                                                    "Sub-task 5b: Make final decision on the correct multiple-choice answer based on prior synthesis and domain validity.",
                                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, making final answer decision, thinking: {thinking5b_final.content}; answer: {answer5b_final.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b_final.content}; answer - {answer5b_final.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b_final, "answer": answer5b_final}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5b_final, answer5b_final, sub_tasks, agents)
    return final_answer, logs
