async def forward_196(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Extract and summarize the given IR and NMR data
    debate_instruction_0 = "Sub-task 1: Extract and summarize the given IR and NMR data to identify functional groups and structural features of Compound X."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    all_thinking0 = []
    all_answer0 = []
    for agent in debate_agents_0:
        thinking0, answer0 = await agent([taskInfo], debate_instruction_0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking0.content}; answer: {answer0.content}")
        all_thinking0.append(thinking0)
        all_answer0.append(answer0)
    sub_tasks.append(f"Sub-task 0 output: thinking - {all_thinking0[-1].content}; answer - {all_answer0[-1].content}")
    logs.append({
        "subtask_id": "subtask_0",
        "instruction": debate_instruction_0,
        "response": {
            "thinking": all_thinking0[-1],
            "answer": all_answer0[-1]
        }
    })
    print("Step 0: ", sub_tasks[-1])

    # Stage 1: Analyze the relationship between functional groups and reaction conditions
    cot_sc_instruction_1 = "Sub-task 1: Analyze the relationship between the identified functional groups and the reaction conditions with red phosphorus and HI."
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    for i in range(N):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo, all_thinking0[-1], all_answer0[-1]], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, all_thinking0[-1], all_answer0[-1]] + possible_thinkings_1 + possible_answers_1, 
                                                 "Sub-task 1: Synthesize and choose the most consistent answer for the relationship analysis.", 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "response": {
            "thinking": thinking1,
            "answer": answer1
        }
    })
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Predict the final product
    debate_instruction_2 = "Sub-task 1: Predict the final product by applying the transformation criteria to the input elements and generate possible variants."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    all_thinking2 = []
    all_answer2 = []
    for agent in debate_agents_2:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction_2, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        all_thinking2.append(thinking2)
        all_answer2.append(answer2)
    sub_tasks.append(f"Sub-task 2 output: thinking - {all_thinking2[-1].content}; answer - {all_answer2[-1].content}")
    logs.append({
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "response": {
            "thinking": all_thinking2[-1],
            "answer": all_answer2[-1]
        }
    })
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Evaluate the predicted variants against the given choices
    cot_instruction_3 = "Sub-task 2: Evaluate the predicted variants against the given choices to identify the most likely final product."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, all_thinking2[-1], all_answer2[-1]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "response": {
            "thinking": thinking3,
            "answer": answer3
        }
    })
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
