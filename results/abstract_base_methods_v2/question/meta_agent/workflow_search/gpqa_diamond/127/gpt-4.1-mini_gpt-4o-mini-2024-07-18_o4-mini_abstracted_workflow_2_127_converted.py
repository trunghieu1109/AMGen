async def forward_127(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1: Translate the given Human P53 amino acid sequence into its corresponding nucleotide (DNA) sequence optimized for expression in E. coli BL21."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, translating Human P53 amino acid sequence to codon-optimized DNA, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    cot_instruction_1b = "Sub-task 2: Analyze and parse each of the four provided plasmid DNA sequences to ensure they are complete, correctly formatted, and represent valid coding sequences."
    N_sc = self.max_sc
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc_1b = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1b,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1b, answer_1b = await cot_agents_1b[i]([taskInfo], cot_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, analyzing plasmid DNA sequences, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
        possible_answers_1b.append(answer_1b.content)
        thinkingmapping_1b[answer_1b.content] = thinking_1b
        answermapping_1b[answer_1b.content] = answer_1b
    answer_1b_content = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking_1b = thinkingmapping_1b[answer_1b_content]
    answer_1b = answermapping_1b[answer_1b_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    cot_reflect_instruction_2 = "Sub-task 3: Translate each plasmid DNA sequence into its corresponding amino acid sequence to verify which plasmid encodes the Human P53 protein sequence provided."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b]
    thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, translating plasmid DNA sequences to amino acid sequences, thinking: {thinking_2.content}; answer: {answer_2.content}")
    for i in range(N_max_2):
        feedback_2, correct_2 = await critic_agent_2([taskInfo, thinking_2, answer_2], "please review the translation of plasmid DNA sequences and verify correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback_2.content}; answer: {correct_2.content}")
        if correct_2.content == "True":
            break
        cot_inputs_2.extend([thinking_2, answer_2, feedback_2])
        thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining translation, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc_3['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_3)
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    debate_instruction_4 = "Sub-task 4: Compare the translated amino acid sequences from each plasmid with the target Human P53 amino acid sequence to identify the exact match."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_2, answer_2], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_2, answer_2] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing translated amino acid sequences, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Make final decision on which plasmid encodes the exact Human P53 protein sequence.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding exact plasmid match, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    cot_instruction_5 = "Sub-task 5: Select the plasmid whose translated amino acid sequence perfectly matches the Human P53 protein sequence, ensuring it is suitable for expression in E. coli BL21."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking_5, answer_5 = await cot_agent_5([taskInfo, thinking_4, answer_4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, selecting suitable plasmid for expression, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Subtask 5 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs
