async def forward_190(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_role = self.debate_role
    max_round = self.max_round

    debate_instruction_1 = "Sub-task 1: Analyze Step 1: Determine the structural changes when the starting material is treated with sodium hydride and benzyl bromide, focusing on the alkylation of the hydroxyl group to form product 1. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_role]

    all_thinking_1 = [[] for _ in range(max_round)]
    all_answer_1 = [[] for _ in range(max_round)]

    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }

    for r in range(max_round):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instruction_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, Sub-task 1, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_1[r].append(thinking1)
            all_answer_1[r].append(answer1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking_1[-1] + all_answer_1[-1], final_instr_1, is_sub_task=True)
    agents.append(f"Final Decision agent Sub-task 1, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_2 = "Sub-task 2: Analyze Step 2: Elucidate the transformation of product 1 upon treatment with p-toluenesulfonyl hydrazide and catalytic HCl, identifying the formation of product 2 (likely a tosyl hydrazone derivative). Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_role]

    all_thinking_2 = [[] for _ in range(max_round)]
    all_answer_2 = [[] for _ in range(max_round)]

    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "Debate"
    }

    for r in range(max_round):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, Sub-task 2, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking_2[r].append(thinking2)
            all_answer_2[r].append(answer2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking_2[-1] + all_answer_2[-1], final_instr_2, is_sub_task=True)
    agents.append(f"Final Decision agent Sub-task 2, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3 = "Sub-task 3: Analyze Step 3: Interpret the reaction of product 2 with n-butyllithium at low temperature followed by aqueous ammonium chloride quench, deducing the structure and functional group changes leading to product 3 (likely via a Shapiro reaction or related mechanism). Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_role]

    all_thinking_3 = [[] for _ in range(max_round)]
    all_answer_3 = [[] for _ in range(max_round)]

    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Debate"
    }

    for r in range(max_round):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, Sub-task 3, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3 = "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking_3[-1] + all_answer_3[-1], final_instr_3, is_sub_task=True)
    agents.append(f"Final Decision agent Sub-task 3, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_4 = "Sub-task 4: Analyze Step 4: Determine the effect of catalytic hydrogenation (Pd/C, H2) on product 3, including reduction of double bonds and removal of benzyl protecting groups, to deduce the structure of product 4. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_role]

    all_thinking_4 = [[] for _ in range(max_round)]
    all_answer_4 = [[] for _ in range(max_round)]

    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "Debate"
    }

    for r in range(max_round):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, Sub-task 4, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3, answer3] + all_thinking_4[-1] + all_answer_4[-1], final_instr_4, is_sub_task=True)
    agents.append(f"Final Decision agent Sub-task 4, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = "Sub-task 5: Integrate the mechanistic insights and structural deductions from all previous steps to evaluate the four given candidate structures and select the correct structure of product 4. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_role]

    all_thinking_5 = [[] for _ in range(max_round)]
    all_answer_5 = [[] for _ in range(max_round)]

    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4, answer4],
        "agent_collaboration": "Debate"
    }

    for r in range(max_round):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, Sub-task 5, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking_5[r].append(thinking5)
            all_answer_5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_5 = "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + all_thinking_5[-1] + all_answer_5[-1], final_instr_5, is_sub_task=True)
    agents.append(f"Final Decision agent Sub-task 5, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
