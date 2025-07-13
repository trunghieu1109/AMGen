async def forward_165(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Extract and summarize model definition with Chain-of-Thought
    cot_instruction = (
        "Sub-task 1: Extract and summarize the model definition from the Lagrangian, "
        "including kinetic terms, field content, quantum numbers, and VEVs x and v."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT",
        "response": {"thinking": thinking1, "answer": answer1}
    })
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 3: Enumerate one-loop contributions with Self-Consistency Chain-of-Thought
    sc_instruction = (
        "Sub-task 3: Enumerate all one-loop contributions to the pseudo-Goldstone H2 mass, "
        "listing bosonic and fermionic loops and their couplings and signs."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    sub_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": sc_instruction,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        t, a = await cot_agents[i]([taskInfo, thinking1, answer1], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {t.content}; answer: {a.content}")
        possible_thinkings.append(t)
        possible_answers.append(a)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instruction = (
        "Sub-task 3 Decision: Given all SC-CoT thoughts and answers, select the consistent list of one-loop contributions for H2 mass."
    )
    thinking3, answer3 = await final_decision_agent(
        [taskInfo, thinking1, answer1] + possible_thinkings + possible_answers,
        decision_instruction, is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({
        **sub_desc_3,
        "response": {"thinking": thinking3, "answer": answer3}
    })
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 4: Assemble effective potential and compute mass correction with Debate
    debate_instr = (
        "Sub-task 4: Assemble the one-loop effective potential contributions and compute δM_h2^2. "
        "Include prefactor 1/(8π²(x²+v²)) and sum over M^4 terms."
    )
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    sub_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                t4, a4 = await agent([taskInfo, thinking3, answer3], debate_instr, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                t4, a4 = await agent(inputs, debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t4.content}; answer: {a4.content}")
            all_thinking4[r].append(t4)
            all_answer4[r].append(a4)
    final_decision_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = (
        "Sub-task 4 Final Decision: Given all debate outputs, provide the assembled expression for M_h2^2."
    )
    thinking4, answer4 = await final_decision_4(
        [taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1],
        final_instr_4, is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append({
        **sub_desc_4,
        "response": {"thinking": thinking4, "answer": answer4}
    })
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs