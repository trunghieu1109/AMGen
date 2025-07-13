async def forward_26(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_0 = (
        "Sub-task 1: Derive the formula relating the total number of sets B to the elements of A, "
        "specifically showing that the total count equals the sum over a in A of 2^(a-1). "
        "Validate this representation with the given total of 2024."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking0, answer0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving formula, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {
        "thinking": thinking0,
        "answer": answer0
    }
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])
    
    debate_instr_1 = (
        "Sub-task 1: Identify the elements of A by expressing 2024 as a sum of distinct powers of two of the form 2^(a-1), "
        "ensuring each corresponds to a positive integer a. Verify the uniqueness and correctness of this decomposition. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                    model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo, thinking0, answer0] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_1[r].append(thinking1)
            all_answer_1[r].append(answer1)
    final_decision_instr_1 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                          model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + all_thinking_1[-1] + all_answer_1[-1], 
                                                    "Sub-task 1: Identify elements of A." + final_decision_instr_1, 
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, round {N_max_1-1}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])
    
    reflect_inst_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2 = (
        "Sub-task 2: Simplify the identified elements of A and compute their sum, which is the required final answer. "
        + reflect_inst_2
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                                  model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking0, answer0, thinking1, answer1]
    subtask_desc_2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", thinking0.content, answer0.content, thinking1.content, answer1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, simplifying and summing elements of A, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_2):
        critic_inst_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], 
                                               "Please review and provide the limitations of provided solutions." + critic_inst_2, 
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining solution, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3 = (
        "Sub-task 3: Aggregate the results from previous subtasks to confirm the sum of elements in A and present the final solution clearly."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    subtask_desc_3 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking0.content, answer0.content, thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, aggregating final results, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
