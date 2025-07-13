async def forward_24(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Rewrite each given logarithmic equation in terms of log2(x), log2(y), and log2(z) by applying logarithm properties to transform the ratios into linear expressions. "
        "Given the equations: log2(x/(yz))=1/2, log2(y/(xz))=1/3, log2(z/(xy))=1/4, express each as a linear equation in variables a=log2(x), b=log2(y), c=log2(z)."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, rewriting logarithmic equations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Express the target expression log2(x^4 y^3 z^2) as a linear combination of log2(x), log2(y), and log2(z) using logarithm properties. "
        "Write it in terms of variables a=log2(x), b=log2(y), c=log2(z)."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, expressing target expression, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Formulate a system of linear equations from the transformed logarithmic equations obtained in stage_0 subtask_1, "
        "representing them in terms of variables a = log2(x), b = log2(y), and c = log2(z)."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, formulating linear system, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Solve the system of linear equations for a, b, and c or find a direct expression for the target linear combination 4a + 3b + 2c using the system. "
        "Use the equations from stage_1.subtask_3 and the target expression from stage_0.subtask_2. "
        "Provide the exact rational value of log2(x^4 y^3 z^2)."
    )
    N = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content, thinking2.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, thinking2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, solving linear system, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent and correct solution for the target expression.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing solution, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Calculate the absolute value of the expression log2(x^4 y^3 z^2) obtained from the solution in stage_1 subtask_4, "
        "ensuring the result is a positive rational number."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "stage_2.subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating absolute value, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_6 = (
        "Sub-task 6: Simplify the absolute value result to a reduced fraction m/n where m and n are relatively prime positive integers. "
        "Provide the simplified fraction explicitly."
    )
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6 = []
    possible_thinkings_6 = []
    subtask_desc6 = {
        "subtask_id": "stage_2.subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, simplifying fraction, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6)
        possible_thinkings_6.append(thinking6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_thinkings_6, "Sub-task 6: Synthesize and choose the most consistent and correct simplified fraction.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing simplified fraction, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    debate_instr_7 = (
        "Sub-task 7: Compute the sum m + n from the reduced fraction obtained in stage_2 subtask_6 and present the final answer. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    subtask_desc7 = {
        "subtask_id": "stage_3.subtask_7",
        "instruction": debate_instr_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6], debate_instr_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6] + all_thinking7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instr_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing final answer, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1], "Sub-task 7: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
