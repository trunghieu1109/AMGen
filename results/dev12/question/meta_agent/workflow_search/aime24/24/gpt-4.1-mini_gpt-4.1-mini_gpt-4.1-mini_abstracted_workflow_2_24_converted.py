async def forward_24(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_stage0 = (
        "Sub-task 1: Rewrite each given logarithmic equation log2(x/(yz))=1/2, log2(y/(xz))=1/3, and log2(z/(xy))=1/4 "
        "into linear equations in terms of a=log2 x, b=log2 y, c=log2 z. "
        "Carefully apply logarithm properties to express each equation as a linear combination of a, b, c. "
        "Avoid solving at this stage; focus only on correct transformation."
    )
    N_sc = self.max_sc
    cot_sc_agents_stage0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage0 = []
    possible_thinkings_stage0 = []
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_stage0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking0, answer0 = await cot_sc_agents_stage0[i]([taskInfo], cot_sc_instruction_stage0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_stage0[i].id}, rewriting logarithmic equations, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_stage0.append(answer0)
        possible_thinkings_stage0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_answers_stage0 + possible_thinkings_stage0, "Sub-task 1: Synthesize and choose the most consistent rewriting of equations." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_stage1 = (
        "Sub-task 1: Solve the system of linear equations obtained from Stage 0 to find explicit values for a=log2 x, b=log2 y, c=log2 z. "
        "Use algebraic methods such as substitution or elimination. Ensure the solution is consistent and respects positivity of x,y,z. "
        "Avoid approximations; keep exact fractional values."
    )
    cot_agent_stage1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_stage1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_stage1([taskInfo, thinking0, answer0], cot_instruction_stage1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage1.id}, solving linear system, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_stage2_1 = (
        "Sub-task 1: Compute log2(x^4 y^3 z^2) using the values of a,b,c found in Stage 1 by applying the logarithm power rule: 4a + 3b + 2c. "
        "Carefully perform the arithmetic to obtain an exact fractional value. Avoid rounding errors."
    )
    cot_agent_stage2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_stage2_1,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2_1, answer2_1 = await cot_agent_stage2_1([taskInfo, thinking1, answer1], cot_instruction_stage2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage2_1.id}, computing log2(x^4 y^3 z^2), thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_sc_instruction_stage2_2 = (
        "Sub-task 2: Calculate the absolute value of the result from subtask_1 in Stage 2. "
        "Confirm the sign and ensure the output is a positive fraction in lowest terms. Avoid skipping the absolute value step or simplifying incorrectly."
    )
    cot_sc_agents_stage2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage2_2 = []
    possible_thinkings_stage2_2 = []
    subtask_desc2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_stage2_2,
        "context": ["user query", thinking2_1.content, answer2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2_2, answer2_2 = await cot_sc_agents_stage2_2[i]([taskInfo, thinking2_1, answer2_1], cot_sc_instruction_stage2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_stage2_2[i].id}, calculating absolute value, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
        possible_answers_stage2_2.append(answer2_2)
        possible_thinkings_stage2_2.append(thinking2_2)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_2, answer2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_stage2_2 + possible_thinkings_stage2_2, "Sub-task 2: Synthesize and choose the most consistent absolute value result.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 3.2: ", sub_tasks[-1])

    debate_instr_stage3 = (
        "Sub-task 1: Express the absolute value |log2(x^4 y^3 z^2)| = m/n as a reduced fraction with relatively prime positive integers m and n. "
        "Then compute and output the sum m + n. Carefully verify the fraction is in simplest form before summation. Avoid errors in fraction reduction or arithmetic. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_debate = self.max_round
    all_thinking_stage3 = [[] for _ in range(N_max_debate)]
    all_answer_stage3 = [[] for _ in range(N_max_debate)]
    subtask_desc3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_stage3,
        "context": ["user query", thinking2_2.content, answer2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_debate):
        for i, agent in enumerate(debate_agents_stage3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2_2, answer2_2], debate_instr_stage3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2_2, answer2_2] + all_thinking_stage3[r-1] + all_answer_stage3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_stage3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, reasoning fraction reduction and sum, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_stage3[r].append(thinking3)
            all_answer_stage3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking_stage3[-1] + all_answer_stage3[-1], "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final output, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
