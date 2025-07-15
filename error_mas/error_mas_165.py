from collections import Counter

async def forward_165(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Extract fields, VEVs, and couplings from the Lagrangian that enter the pseudo-Goldstone mass calculation."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinking1 = []
    possible_answer1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, extracting fields and couplings, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinking1.append(thinking)
        possible_answer1.append(answer)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinking1 + possible_answer1, "Sub-task 1: Synthesize and choose the most consistent extraction based on the above thoughts and answers.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Derive the general one-loop Coleman-Weinberg mass expression for a pseudo-Goldstone boson in terms of the VEVs x and v."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinking2 = []
    possible_answer2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, deriving the Coleman-Weinberg formula, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinking2.append(thinking)
        possible_answer2.append(answer)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinking2 + possible_answer2, "Sub-task 2: Synthesize and choose the most consistent Coleman-Weinberg mass expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 3: Identify and list all particle contributions—h1, W, Z, H±, H0, A0, top quark, heavy Ni—with loop-induced signs and multiplicity coefficients α_i."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinking3 = []
    possible_answer3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, listing particle contributions, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinking3.append(thinking)
        possible_answer3.append(answer)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + possible_thinking3 + possible_answer3, "Sub-task 3: Synthesize and choose the most consistent list of particle contributions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = "Sub-task 4: Compare each of the four provided formula choices against the derived structure: check the prefactor, which masses appear, and whether boson loops enter with + and fermion loops with – signs." + debate_instr
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking, answer = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking4[r].append(thinking)
            all_answer4[r].append(answer)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instruction4 = "Sub-task 4: given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], final_instruction4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 5: Select the choice that exactly matches the derived one-loop mass formula for Mh2^2 and provide a concise justification."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinking5 = []
    possible_answer5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo, thinking4, answer4], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, selecting matching formula choice, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinking5.append(thinking)
        possible_answer5.append(answer)
    final_decision_agent5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo, thinking4, answer4] + possible_thinking5 + possible_answer5, "Sub-task 5: Synthesize and choose the final correct formula choice with justification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer, logs = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs