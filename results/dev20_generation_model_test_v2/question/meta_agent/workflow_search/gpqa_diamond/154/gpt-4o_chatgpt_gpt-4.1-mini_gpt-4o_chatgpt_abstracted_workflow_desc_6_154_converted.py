async def forward_154(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs =  []
    
    cot_instruction_0_1 = "Sub-task 1: Extract and summarize the given information about the operators, state, and eigenvalues."
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing given information, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {
        "thinking": thinking_0_1,
        "answer": answer_0_1
    }
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_0_2 = "Sub-task 2: Analyze the relationships between the operators and the state, focusing on the implications of the system being in an eigenstate of P_x."
    N_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", "thinking of subtask 0.1", "answer of subtask 0.1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_0_2):
        thinking_0_2, answer_0_2 = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, analyzing relationships, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2)
        possible_thinkings_0_2.append(thinking_0_2)
    final_instr_0_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the relationships."
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_0_2 + possible_answers_0_2, 
                                                 "Sub-task 0.2: Synthesize and choose the most consistent answer for relationships" + final_instr_0_2, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {
        "thinking": thinking_0_2,
        "answer": answer_0_2
    }
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_1_1 = "Sub-task 1: Compute the expectation value of P_z in the given state."
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", "thinking of subtask 0.2", "answer of subtask 0.2"],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, computing expectation value of P_z, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {
        "thinking": thinking_1_1,
        "answer": answer_1_1
    }
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_1_2 = "Sub-task 2: Compute the expectation value of P_z squared in the given state."
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", "thinking of subtask 1.1", "answer of subtask 1.1"],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, computing expectation value of P_z squared, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {
        "thinking": thinking_1_2,
        "answer": answer_1_2
    }
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_2_1 = "Sub-task 1: Calculate the variance of P_z using the expectation values obtained."
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", "thinking of subtask 1.2", "answer of subtask 1.2"],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_2, answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, calculating variance of P_z, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {
        "thinking": thinking_2_1,
        "answer": answer_2_1
    }
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_2_2 = "Sub-task 2: Derive the uncertainty ΔP_z from the variance."
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "subtask_2_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", "thinking of subtask 2.1", "answer of subtask 2.1"],
        "agent_collaboration": "CoT"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, deriving uncertainty ΔP_z, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {
        "thinking": thinking_2_2,
        "answer": answer_2_2
    }
    logs.append(subtask_desc_2_2)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs